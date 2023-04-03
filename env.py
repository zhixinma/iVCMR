import torch
import numpy as np
import networkx as nx
from typing import Union
import random
from torch.nn.utils.rnn import pad_sequence
from data import LMDBFile
from load import load_graph
from load import load_concept_prob
from util import NO_CHOICE_CONCEPT


class IVCMLEnv(object):
    def __init__(self, opts, concepts, shot_ids):
        # Configuration
        self.opts = opts
        self.concepts = concepts
        self.shot_ids = shot_ids
        self.concepts_to_idx = {tok: i_tok for i_tok, tok in enumerate(concepts)}
        self.concept_size = len(concepts)
        self.gamma = opts.gamma  # discount factor for state value
        self.gae_lambda = opts.gae_lambda  # discount factor for gae
        self.phi = opts.phi  # punishment factor for time step
        self.alpha = opts.alpha  # discount factor for node observation
        self.max_distance = 5
        self.reward_signal = opts.reward_signal
        self.sample_action = opts.sample_action
        self.max_action_space = opts.max_action_space
        self.wo_replacement = opts.wo_replacement
        self.feature_type = opts.feature_type

        # path
        self.feature_root = opts.feature_root
        self.dataset = opts.dataset
        self.prob_path = f"{self.feature_root}/{self.dataset}_dual_task_concept_p3_2048.h5"

        # session information recording
        # batch index (agent id) -> visited node list
        self.visited_dict = []

        # action_type determine the constitution of neighbors
        self.action_type = opts.action_type
        self.estimate_gold = opts.estimate_gold

        # if drop_graph is True, node feature will be itself instead of aggregated surrounding feature
        # and edge feature will not be considered
        self.drop_graph = self.opts.drop_graph
        self.training = True

        # record
        self.missed_feature_id = []
        self.c_missed_node = 0

        # load data readers
        self.data_root = opts.data_root
        tag_graph_param = "wc_%.2f_ws_%.2f_ths_%.2f_thx_%.2f" % (self.opts.weight_concept, self.opts.weight_subtitle, self.opts.edge_threshold_single, self.opts.edge_threshold_cross)
        if self.opts.use_balanced_graph:
            tag_graph_param += f"_top_{self.opts.k_ei}"
        tag_trans = "clip_level_env_trans_%s" % tag_graph_param
        env_trans_reader = LMDBFile(self.data_root, self.feature_type, self.opts.split, tag=tag_trans, encode_method="ndarray", readonly=True)
        tag_moment_dist = "clip_level_moment_distance_%s" % tag_graph_param
        moment_distance_reader = LMDBFile(self.data_root, self.feature_type, self.opts.split, tag=tag_moment_dist, encode_method="ndarray", readonly=True)
        tag_clip_feat = "clip_level_unit_fea"
        hero_clip_fea_reader = LMDBFile(self.data_root, self.feature_type, self.opts.split, tag=tag_clip_feat, encode_method="ndarray", readonly=True)
        tag_n6 = f"graph_neighbors_in_k_6_{tag_graph_param}"
        n6_reader = LMDBFile(self.data_root, self.feature_type, self.opts.split, tag=tag_n6, encode_method="json", readonly=True)
        self.trans_reader = env_trans_reader
        self.moment_distance_reader = moment_distance_reader
        self.hero_clip_fea_reader = hero_clip_fea_reader
        self.n6_reader = n6_reader

        # load graph and concept probability
        nk_reader = None
        if self.opts.action_type == "nk":
            tag_nk = f"graph_neighbors_in_k_{self.opts.k_for_nk}_{tag_graph_param}"
            nk_reader = LMDBFile(self.data_root, self.feature_type, self.opts.split, tag=tag_nk, encode_method="json", readonly=True)
        graph = load_graph(nk_reader, self.opts)
        concept_prob = load_concept_prob(self.prob_path)
        self.node_concept_prob = concept_prob
        self.graph = graph

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False

    def move(self, states: torch.Tensor, move_indices: torch.Tensor):
        device = states.device
        batch_size = states.shape[0]
        states_next = []
        for i_sample in range(batch_size):
            state_id = states[i_sample].item()
            move_idx = move_indices[i_sample].item()
            state_next = self.__move__(state_id, move_idx, i_sample)
            states_next.append(state_next)
        states_next = torch.tensor(states_next, device=device, dtype=torch.long)
        return states_next

    def get_advantage(self, states_steps_with_last: torch.Tensor, value_pred: torch.Tensor, last_value: torch.Tensor, reward: torch.Tensor, active_mask: torch.Tensor):
        """
        :param states_steps_with_last: [B, S+1], where S denotes the number of steps
        :param value_pred:
        :param last_value:
        :param reward: reward of each step
        :param active_mask: indicate does the agent active
        :return:
        """
        device = states_steps_with_last.device
        batch_size = states_steps_with_last.shape[0]
        num_steps = states_steps_with_last.shape[1] - 1
        value_with_last = torch.cat([value_pred, last_value.unsqueeze(dim=1)], dim=1)

        # Use Generalized Advantage Estimation (GAE)
        # (https://arxiv.org/abs/1506.02438) to compute the advantage.
        advantage = torch.zeros((batch_size, num_steps), device=device)
        last_gae_lam = 0
        for i_step in reversed(range(num_steps)):
            reward[:, i_step] = reward[:, i_step] - self.phi*i_step
            next_non_terminal = active_mask[:, i_step + 1]
            next_values = value_with_last[:, i_step + 1]
            delta = reward[:, i_step] + self.gamma * next_values * next_non_terminal - value_pred[:, i_step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantage[:, i_step] = last_gae_lam

        returns = advantage + value_pred
        return advantage, returns

    def get_reward(self, states_steps_with_last: torch.Tensor, target_ids: torch.Tensor, query_ids: torch.Tensor):
        def __reward_by_step__():
            # calculate distance first
            distance_steps = self.get_all_steps_distance_to_moment(states_steps_with_last, query_ids)

            # calculate reward by comparing distance
            # Note: default is zero
            init_dist = distance_steps[:, :-1]
            pred_dist = distance_steps[:, 1:]
            reward_abs_value = 1 / torch.exp2(pred_dist)
            reward_batch = torch.zeros((batch_size, num_steps), device=device)
            good_coord = torch.gt(init_dist, pred_dist).nonzero(as_tuple=True)
            bad_coord = torch.lt(init_dist, pred_dist).nonzero(as_tuple=True)
            reward_batch[good_coord] = reward_abs_value[good_coord] * 1
            # reward_batch[bad_coord] = reward_abs_value[bad_coord] * -1
            reward_batch[bad_coord] = - 0.5  # constant punishment
            return reward_batch

        def __reward_by_path__():
            pred_state_steps = states_steps_with_last[:, 1:]
            reward_batch = sum([pred_state_steps == target_ids[:, i].unsqueeze(dim=-1) for i in range(target_ids.shape[1])]).to(torch.float32)
            return reward_batch

        device = states_steps_with_last.device
        batch_size = states_steps_with_last.shape[0]
        num_steps = states_steps_with_last.shape[1] - 1

        if self.reward_signal == "step":
            reward = __reward_by_step__()
        elif self.reward_signal == "path":
            reward = __reward_by_path__()
        else:
            assert False, "Error: Bad reward signal."

        return reward

    def get_all_steps_distance_to_moment(self, states_steps: torch.Tensor, query_ids: torch.Tensor):
        device = states_steps.device
        batch_size = states_steps.shape[0]

        # calculate distance first
        distance_steps = torch.zeros_like(states_steps, device=device)
        for i_sample in range(batch_size):
            distance_map = self.__moment_distance__(query_ids[i_sample].item())
            state_ids_agent = [states_steps[i_sample]]
            distance_steps[i_sample] = distance_map[state_ids_agent]
        return distance_steps

    def get_gold_transition(self, query_ids: torch.Tensor):
        vec_reader = self.trans_reader
        return self.__get_query_related_vec__(query_ids, vec_reader)

    def get_mom_distance(self, query_ids: torch.Tensor):
        vec_reader = self.moment_distance_reader
        return self.__get_query_related_vec__(query_ids, vec_reader)

    @staticmethod
    def __get_query_related_vec__(query_ids: torch.Tensor, vec_reader):
        device = query_ids.device
        batch_size = query_ids.shape[0]
        vec_batch = []
        for i in range(batch_size):
            query_id = query_ids[i].item()
            vec = vec_reader[str(query_id)].to(device)
            vec_batch.append(vec)
        return torch.stack(vec_batch, dim=0)

    def get_gold_bias(self, states: torch.Tensor, trans_batch: torch.Tensor, targets_ids: torch.Tensor):
        """
        :param states: [B]
        :param trans_batch: [B, V] where V denotes the node size
        :param targets_ids:
        :return:
        """

        batch_size = states.shape[0]
        biases = []
        for i_sample in range(batch_size):
            state_id = states[i_sample].item()
            trans = trans_batch[i_sample]
            next_state_gold = int(trans[state_id].item())
            targets = [i for i in targets_ids[i_sample].tolist() if i > 0]

            if state_id in targets:
                bias_idx = -1  # will trigger error if used to calculate loss
            else:
                actions = self.__actions__(state_id, i_sample)
                try:
                    bias_idx = actions.index(next_state_gold)
                except ValueError:
                    bias_idx = -1

            biases.append(bias_idx)
        return torch.tensor(biases, device=states.device, dtype=torch.long)

    def get_nk_gold_bias(self, states: torch.Tensor, d_mom_batch: torch.Tensor, targets_batch: torch.Tensor):
        """
        :param states: [B]
        :param d_mom_batch: [B, V] where V denotes the node size
        :param targets_batch: [B, t_max] where t_max denotes maximum target size
        :return:
        """

        batch_size = states.shape[0]
        biases = []
        for i_sample in range(batch_size):
            state_id = states[i_sample].item()
            target_ids = targets_batch[i_sample].tolist()
            d_mom = d_mom_batch[i_sample]

            bias_idx = -1
            if state_id not in target_ids:  # terminated agent
                actions = self.__actions__(state_id, i_sample)
                # target is in action space
                for t_id in target_ids:
                    if t_id in actions:
                        bias_idx = actions.index(t_id)
                        break
                if bias_idx == -1 and self.estimate_gold:
                    # target is not in action space
                    # chose the nearest one
                    ds = d_mom[actions]
                    bias_indices = (ds == ds.min()).nonzero().squeeze(-1).tolist()
                    bias_idx = random.choice(bias_indices)
            biases.append(bias_idx)

        return torch.tensor(biases, device=states.device, dtype=torch.long)

    def get_node_observe_feature(self, states: torch.Tensor):
        device = states.device
        states = states.tolist()
        obs_feat = []
        for state_id in states:
            f = self.__node_feat__(state_id)
            obs_feat.append(f)
        obs_feat = torch.stack(obs_feat, dim=0)
        return obs_feat.to(device=device)

    def __moment_distance__(self, query_id: int):
        if self.moment_distance_reader is not None:
            return self.moment_distance_reader[str(query_id)]

    def __nei_w_feat__(self, state_id: int, i_sample: int):
        actions = self.__actions__(state_id, i_sample)
        w_feat = []
        for nei_id in actions:
            f = self.__node_feat__(nei_id)
            w_feat.append(f)
        w_feat = torch.stack(w_feat, dim=0) if w_feat else torch.zeros([1, 768])  # when sample without replacement
        return w_feat

    def __node_feat__(self, node_id: int):
        f = self.hero_clip_fea_reader[self.shot_ids[node_id]]
        if f is None:
            f = torch.zeros(1, 768)
            # print("Missed %d feature" % node_id)
            self.c_missed_node += 1
            if node_id not in self.missed_feature_id:
                self.missed_feature_id.append(node_id)
        f = f.squeeze(0)
        return f

    def __move__(self, state_id: int, action_idx: int, i_sample: int):
        actions = self.__actions__(state_id, i_sample)
        next_state = actions[action_idx] if actions else state_id
        return next_state

    def __actions__(self, state_id: int, agent_id: int):
        """
        :param state_id:
        :param agent_id: sample index within a batch
        :return:
        """
        if self.action_type == "n01":
            actions = [state_id] + list(self.graph.adj[state_id].keys())
        elif self.action_type == "n1":
            actions = list(self.graph.adj[state_id].keys())
        elif self.action_type in ["nk", "knn"]:
            actions = self.graph.nodes[state_id]["ns"]
        else:
            assert False

        # record visited nodes
        if self.wo_replacement:
            actions = [i for i in actions if i not in self.visited_dict[agent_id]]

        # sample action space
        if self.sample_action and len(actions) > self.max_action_space:
            random.seed(42)
            actions = random.sample(actions, self.max_action_space)
        return actions

    def __action_num__(self, state_id: int, i_sample: int):
        return len(self.__actions__(state_id, i_sample))

    def __nei_concept_diff__(self, state_id: int, i_sample: int):
        actions = self.__actions__(state_id, i_sample)
        if len(actions) != 0:
            nei_concept_diff = self.node_concept_prob[actions] - self.node_concept_prob[state_id]
            nei_concept_diff = torch.from_numpy(nei_concept_diff)
        else:
            nei_concept_diff = torch.zeros((1, self.concept_size), dtype=torch.float32)
        return nei_concept_diff

    def __tar_concept_diff__(self, state_id: int, target_id: Union[int, list]):
        if isinstance(target_id, int):
            p_tgt = self.node_concept_prob[target_id]
        elif isinstance(target_id, list):
            if len(target_id):
                p_tgt = sum([self.node_concept_prob[i] for i in target_id]) / len(target_id)
            else:
                p_tgt = self.node_concept_prob[state_id]
                print("Error: No target is found.")
        else:
            print("Unsupported target type:", type(target_id))
            raise NotImplementedError

        concept_diff = self.node_concept_prob[state_id] - p_tgt  # negative means missing in current moment
        concept_diff_abs = np.absolute(concept_diff)
        norm = concept_diff_abs.sum().item()
        indicator_concept = torch.zeros(self.concept_size, dtype=torch.float32)

        if norm:  # not zero
            sample_prob = concept_diff_abs / norm
            if not self.training or random.random() > 0.1:
                choice = sample_prob.argmax()
            else:
                choice = np.random.choice(self.concept_size, p=sample_prob)
            sign = np.sign(concept_diff[choice])
            indicator_concept[choice] = 1 * sign
            choice = choice * int(sign)
        else:
            choice = NO_CHOICE_CONCEPT

        return indicator_concept, choice

    # Task specific functions
    def get_neighbor_observe_feature(self, states: torch.Tensor):
        device = states.device
        batch_size = states.shape[0]
        nei_observe_weight_batch = []
        for i_sample in range(batch_size):
            state_id = states[i_sample].item()
            weights = self.__nei_w_feat__(state_id, i_sample).to(device)
            nei_observe_weight_batch.append(weights)
        nei_observe_weight_batch = pad_sequence(nei_observe_weight_batch, batch_first=True)
        return nei_observe_weight_batch

    def get_neighbor_mask(self, states: torch.Tensor):
        device = states.device
        batch_size = states.shape[0]
        nei_mask_batch = []
        for i_sample in range(batch_size):
            state_id = states[i_sample].item()
            mask = torch.ones((self.__action_num__(state_id, i_sample)), device=device, dtype=torch.float32)
            nei_mask_batch.append(mask)
        nei_mask_batch = pad_sequence(nei_mask_batch, batch_first=True)
        return nei_mask_batch

    def get_neighbor_concept_diff(self, states: torch.Tensor):
        device = states.device
        nei_concept_diff = []
        for i_sample in range(states.shape[0]):
            state_id = states[i_sample].item()
            nei_concept_diff.append(self.__nei_concept_diff__(state_id, i_sample).to(device))
        nei_concept_diff = pad_sequence(nei_concept_diff, batch_first=True)
        return nei_concept_diff

    def get_target_concept_diff(self, state: torch.Tensor, targets_ids: torch.Tensor):
        device = state.device
        batch_size = state.shape[0]
        indicator_batch = []
        choices = []
        for i in range(batch_size):
            state_id = state[i].item()
            target_ids = [i for i in targets_ids[i].tolist() if i > 0]
            indicator, choice = self.__tar_concept_diff__(state_id, target_ids)
            indicator_batch.append(indicator.to(device))
            choices.append(choice)
        indicator_batch = torch.stack(indicator_batch, dim=0)
        choices = torch.tensor(choices)
        return indicator_batch, choices

    # trajectory recorder
    def init_nodes_record(self, batch_size: int):
        """ batch level record
        """
        self.visited_dict = [[] for _ in range(batch_size)]

    def record_visited_nodes(self, states: torch.Tensor):
        # don't consider whether the agent is terminated or not
        for i_sample, state_id in enumerate(states.tolist()):
            self.visited_dict[i_sample].append(state_id)

    def clear_cache(self):
        """ clear epoch level cache
        """
        self.missed_feature_id = []
        self.c_missed_node = 0

    def summary(self):
        print("Missed feature id:", end=" ")
        print(self.missed_feature_id)
        print("Missed feature count: %d" % self.c_missed_node)

    def __str__(self):
        return "Graph: %s\n" % nx.info(self.graph) + \
               "Concept Size: %d\n" % self.concept_size + \
               "Alpha: %f\n" % self.alpha

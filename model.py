import torch
from torch import nn
from env import IVCMLEnv
import random
from torch.nn.utils.rnn import pad_sequence
from model_vr import RetrievalBaseModel
from collections import OrderedDict
from datetime import datetime


class DeepQNetIVCML(RetrievalBaseModel):
    def __init__(self, env: IVCMLEnv, d_fea, d_concept_en, d_concept_de, infer_steps, opts):
        super(DeepQNetIVCML, self).__init__(opts, d_fea, d_concept_en, d_concept_de)
        self.env = env
        self.time_by_step = []
        self.time_by_query = []

        # Configuration
        self.infer_steps = infer_steps
        self.no_query_update = opts.no_query_update
        self.p_dropout = 0.9
        self.training = False
        self.transition = opts.transition
        self.cur_step = 0
        self.random_policy = opts.random_policy
        self.no_edge = opts.drop_graph
        self.no_feedback = opts.no_feedback
        self.pad_probability = -1e10
        self.concept_size = self.env.concept_size

        # RL model
        self.concept_emb = nn.Linear(self.concept_size, d_fea)
        # visual
        self.num_feat_update = 2 - int(self.no_edge)
        if self.num_feat_update == 1:
            self.neighbor_shot_feat_mlp = self.visual_feat_mlp
        else:
            self.neighbor_shot_feat_mlp = nn.Linear(self.d_fea * self.num_feat_update, self.d_fea)
        # GRU
        num_input_gru = 2 - int(self.no_feedback)
        self.state_upd_gru = nn.GRUCell(d_fea*num_input_gru, self.d_fea)
        # textual
        textual_input_size = 2  # state feature + query feature
        self.value_net = nn.Sequential(OrderedDict([
            ('mlp1', nn.Linear(self.d_fea * textual_input_size, self.d_fea)),
            ('dropout', nn.Dropout()),
            ('relu', nn.ReLU()),
            ('mlp2', nn.Linear(self.d_fea, 1))
        ]))

        # Similarity
        self.cosine = nn.CosineSimilarity(dim=2)

        # Interaction
        self.browse_history_stack = []

    def forward(self, input_ids_bert, input_mask_bert, segment_ids_bert, tok_ids, bow_vec, start_ids, query_ids, targets_ids):
        """
        :param input_ids_bert: [B, S], where S denotes the query length
        :param input_mask_bert: [B, S], where S denotes the query length
        :param segment_ids_bert: [B, S], where S denotes the query length
        :param tok_ids: [B, S], where S denotes the query length
        :param bow_vec: [B, S], where S denotes the query length
        :param start_ids: [B]
        :param query_ids: [B]
        :param targets_ids: [B, T] where T denotes the maximum number of targets
        :return:
        """
        query_feat = self.query_encode(input_ids_bert, input_mask_bert, segment_ids_bert, tok_ids, bow_vec)
        outputs = self.move(query_feat, start_ids, query_ids, targets_ids)
        return outputs

    # simulated
    def move(self, query_feat: torch.Tensor, start_ids: torch.Tensor, query_ids: torch.Tensor, targets_ids: torch.Tensor):
        seq_len = self.infer_steps
        q_0 = query_feat
        final_pred = torch.zeros_like(query_ids) - 1
        final_mask = torch.zeros_like(query_ids, dtype=torch.bool)
        trans_gold = self.env.get_gold_transition(query_ids)
        mom_distance = self.env.get_mom_distance(query_ids)

        policy_pred, policy_gold = [], []
        value_pred = []   # gold value will be calculated later
        agent_mask, policy_mask = [], []
        states_steps = []
        concept_steps = []

        shot_ids = start_ids
        last_value = None
        q_t = q_0

        # Initialize list to record visited nodes
        self.env.init_nodes_record(start_ids.shape[0])
        st = datetime.now()
        _st = st
        for i_step in range(seq_len):
            self.cur_step = i_step
            is_last_it = (i_step == seq_len-1)

            # record visited nodes
            self.env.record_visited_nodes(shot_ids)

            if self.random_policy:
                policy_pred_step, value_pred_step = self.random_step(shot_ids)
            else:
                # Generate policy, value and update query representation
                policy_pred_step, value_pred_step, q_t, feedback_concepts = self.__step__(q_t, q_0, shot_ids, targets_ids)
                concept_steps.append(feedback_concepts)

            if self.env.action_type in ["n01", "n1"]:
                move_gold_step = self.env.get_gold_bias(shot_ids, trans_gold, targets_ids)
            elif self.env.action_type in ["nk"]:
                move_gold_step = self.env.get_nk_gold_bias(shot_ids, mom_distance, targets_ids)
            else:  # have no gold trans
                move_gold_step = torch.zeros_like(shot_ids) - 1

            policy_pred.append(policy_pred_step)
            policy_gold.append(move_gold_step)
            value_pred.append(value_pred_step)
            states_steps.append(shot_ids)
            agent_mask.append(torch.logical_not(final_mask))  # 1 if agent is active

            # transfer to next shot_ids
            states_next, neighbor_mask_current_step = self.state_transfer(shot_ids, policy_pred_step)
            policy_mask.append(neighbor_mask_current_step)

            # update agent status
            # Mask of ending (active) agents
            final_mask, final_pred = \
                self.state_update(states_next, targets_ids, final_mask, final_pred, is_last_it)

            # update shot_ids and query feature
            del shot_ids
            shot_ids = states_next

            if is_last_it:
                if self.random_policy:
                    _, last_value = self.random_step(shot_ids)
                else:
                    _, last_value, _, feedback_concepts = self.__step__(q_t, q_0, shot_ids, targets_ids)
                    concept_steps.append(feedback_concepts)

                states_steps.append(states_next)  # states with last step
                agent_mask.append(torch.logical_not(final_mask))  # 1 if agent is active

            _seconds = (datetime.now() - _st).total_seconds()
            _st = datetime.now()
            self.time_by_step.append(_seconds)

        seconds = (datetime.now() - st).total_seconds()
        self.time_by_query.append(seconds)

        sequence_dim = 1
        policy_pred = pad_sequence([i.T for i in policy_pred], batch_first=False, padding_value=self.pad_probability).transpose(0, 2)
        policy_gold = torch.stack(policy_gold, dim=sequence_dim)
        value_pred = [i if i.shape != torch.Size([]) else i.unsqueeze(dim=0) for i in value_pred]
        last_value = last_value if last_value.shape != torch.Size([]) else last_value.unsqueeze(dim=0)
        value_pred = torch.stack(value_pred, dim=sequence_dim)
        agent_mask = torch.stack(agent_mask, dim=sequence_dim)
        policy_mask = pad_sequence([i.T for i in policy_mask], batch_first=False, padding_value=0).transpose(0, 2)
        states_steps = torch.stack(states_steps, dim=sequence_dim)
        if not self.random_policy:
            concept_steps = torch.stack(concept_steps, dim=sequence_dim)

        # calculate reward and
        reward = self.env.get_reward(states_steps, targets_ids, query_ids)
        advantage, returns = self.env.get_advantage(states_steps, value_pred, last_value, reward, agent_mask)

        return policy_pred, value_pred, last_value, policy_gold, advantage, returns, policy_mask, \
            agent_mask, states_steps, concept_steps, targets_ids, query_ids

    def __step__(self, q_t, q_0, states, target_ids):
        state_feat = self.env.get_node_observe_feature(states)
        neighbor_feat = self.env.get_neighbor_observe_feature(states)

        feedback_vec, edge_concept = None, None
        if not self.no_edge:
            edge_concept = self.env.get_neighbor_concept_diff(states)

        feedback_concepts = torch.tensor([])
        if not self.no_feedback:
            feedback_vec, feedback_concepts = self.env.get_target_concept_diff(states, target_ids)  # sample concept

        if not self.no_query_update:
            # update query with feedback
            q_t = self.query_update(q_t, q_0, state_feat, feedback_vec)

        # predict policy and value
        nei_pred_policy = self.cosine_policy(q_t, neighbor_feat, edge_concept)
        nei_pred_value = self.pred_value(q_t, state_feat)
        return nei_pred_policy, nei_pred_value, q_t, feedback_concepts

    def state_transfer(self, states_current, nei_policy_pred):
        neighbor_mask_current = self.env.get_neighbor_mask(states_current)
        padding_coo = (1 - neighbor_mask_current).nonzero(as_tuple=True)
        nei_policy_pred[padding_coo] = self.pad_probability
        policy_idx_pred = torch.argmax(nei_policy_pred, dim=-1, keepdim=False)

        policy_pred_next_states = self.env.move(states_current, policy_idx_pred)
        states_next = policy_pred_next_states
        return states_next, neighbor_mask_current

    def random_step(self, states):
        device = states.device
        nei_size = [self.env.__action_num__(state_id, i_sample) for i_sample, state_id in enumerate(states.tolist())]
        max_nei_size = max(nei_size)
        batch_size = states.shape[0]
        nei_pred_policy = torch.zeros([batch_size, max_nei_size], device=device)
        for i_sample in range(batch_size):
            n_candidates = nei_size[i_sample]-1
            if n_candidates >= 0:
                rand_idx = random.randint(0, n_candidates)
                nei_pred_policy[i_sample, rand_idx] = 1
        nei_pred_reward = torch.zeros(batch_size, device=device)
        return nei_pred_policy, nei_pred_reward

    def pred_value(self, query_feat, state_feat):
        state_feat = torch.cat((query_feat, state_feat), dim=-1)
        state_value_pred = self.value_net(state_feat).squeeze()
        return state_value_pred

    def cosine_policy(self, query_feat, nei_feat, edge_concept):
        """
        :param query_feat: [B, D]
        :param nei_feat: [B, N, D]
        :param edge_concept: [B, N, D]
        :return:
        """
        # Compose input feature
        if self.no_edge:
            nei_feat = nei_feat
        else:
            edge_concept = self.concept_emb(edge_concept)
            nei_feat = torch.cat((nei_feat, edge_concept), dim=-1)

        nei_feat = self.neighbor_shot_feat_mlp(nei_feat)
        cosine_sim = self.cosine(query_feat.unsqueeze(dim=1), nei_feat)  # [B, 1, D], [B, N, D]
        return cosine_sim

    def query_update(self, q_t, q_0, s, f):
        """
        :param q_t: query hidden feature
        :param q_0: initial query feature generated from BERT
        :param s: state feature generated from HERO
        :param f: feedback concept difference
        :return: updated query hidden feature
        """
        if self.no_feedback:
            i = s
        else:
            f = self.concept_emb(f)
            i = torch.cat((s, f), dim=-1)

        q_t_plus_1 = self.state_upd_gru(i, q_t) + q_0
        return q_t_plus_1

    @staticmethod
    def state_update(states_next_pred: torch.Tensor,
                     targets_ids: torch.Tensor,
                     final_mask: torch.Tensor,
                     final_pred: torch.Tensor,
                     is_last_it: bool):

        # update status (end or not)
        ends = (states_next_pred.unsqueeze(dim=-1) == targets_ids).sum(dim=-1)
        active = torch.logical_not(final_mask)
        new_ends = torch.logical_and(active, ends)

        new_ends_pos = new_ends.nonzero(as_tuple=True)
        final_pred[new_ends_pos] = states_next_pred[new_ends_pos]
        final_mask = final_mask | new_ends

        # update final pred
        if is_last_it:
            not_success = torch.logical_not(final_mask)
            not_success_pos = not_success.nonzero(as_tuple=True)
            final_pred[not_success_pos] = states_next_pred[not_success_pos]
        return final_mask, final_pred

import os.path
import random
from pprint import pprint
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch import optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from load import load_video_id
from data import LMDBFile
from load import load_query_input
from load import load_concepts
from model import DeepQNetIVCML
from env import IVCMLEnv
from util import cuda
from vocab import Vocab
from loss import TripletLoss

from collections import OrderedDict
from util import mean
from util import flatten
import json
from util import MAX_MOMENT_DISTANCE


class Trainer(object):
    def __init__(self, opts):
        super(Trainer, self).__init__()
        self.opts = opts
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.feature_type = opts.feature_type

        # Load concepts
        self.data_root = opts.data_root
        self.feature_root = opts.feature_root
        self.dataset = opts.dataset
        concept_list_path = os.path.join(self.feature_root, f"{self.dataset}_dual_task_emb_concept_p3_2048_concept_bank.txt")
        concepts, concept_size = load_concepts(concept_list_path)
        # Load video id
        video_id_path = os.path.join(self.feature_root, f"{self.dataset}_dual_task_emb_concept_p3_2048_vid_id.txt")
        shot_ids, shot_id_size = load_video_id(video_id_path)
        self.vocab = Vocab(concepts, opts.stem)

        self.env = IVCMLEnv(self.opts, concepts, shot_ids)
        self.minimum_distance = 1
        self.maximum_distance = 4
        self.infer_steps = self.opts.infer_steps
        self.dqn = DeepQNetIVCML(env=self.env,
                                 d_fea=768,
                                 d_concept_en=self.vocab.concept_size_en,
                                 d_concept_de=self.vocab.concept_size_de,
                                 infer_steps=self.infer_steps,
                                 opts=opts)
        self.dqn.to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=opts.init_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)  # step based on conditions

        self.checkpoint = None
        self.is_inference = False
        self.cur_data_size = 0

        # display training info
        pprint(vars(opts))

        # General configuration
        self.training_prefix = opts.training_prefix
        self.judge_metric = "acc"
        self.best_model_path = "%s/%s_best_%s" % (opts.ckpt_root, self.training_prefix, self.judge_metric)
        print("Best checkpoint will be stored at: %s." % self.best_model_path)

        # Training configuration
        self.num_epoch = 200
        self.batch_size = opts.batch_size  # 4 for nk3
        self.evaluate_period = 1
        self.ent_coef = opts.ent_coef
        self.vf_coef = opts.vf_coef
        self.metric_cats = ["acc", "recall", "precision", "f1"]
        self.early_stop_epoch_num = opts.early_stop_epoch_num
        self.transition = opts.transition
        self.normalize_advantage = True
        self.loss_weight = [float(w) for w in opts.loss_weight.split(",")]
        self.loss_func = opts.loss_func.split(",")
        for func in self.loss_func:
            assert func in ["a2c", "ce", "triplet"], "Error: Unsupported loss function: %s" % func
            print("%s weight: %f" % (func, self.loss_weight[self.loss_func.index(func)]))
        self.select_by = opts.select_by
        self.max_grad_norm = 2

        # Inference configuration
        self.step_used = {}
        self.traj_collection = {}
        self.traj_degree = []
        self.c_recorded_sample = 0

        # recorder
        self.ce_loss_epoch = []
        self.triplet_loss_epoch = []
        self.policy_loss_epoch = []
        self.value_loss_epoch = []
        self.entropy_loss_epoch = []
        self.c_covered_step_nk, self.c_out_step_nk = 0, 0

        # global variables
        self.no_improve_epoch_num = 0
        self.is_early_stop = False
        self.no_decay_epoch_num = 0
        self.is_decay = False
        self.best_metrics = {"acc": 0, "recall": 0, "precision": 0, "f1": 0}
        self.i_epoch = 0

        # Loss functions
        self.ce_loss_func = torch.nn.CrossEntropyLoss()
        self.mse_loss_func = torch.nn.MSELoss()
        self.triplet_loss_func = TripletLoss(margin=self.opts.margin,
                                             measure="cosine",
                                             max_violation=False,
                                             cost_style="all",
                                             direction="all",
                                             seq_match=False,
                                             pos_loss_term=False)

    def run(self):
        input_reader = self.load_input_reader()
        if self.opts.mode in ["train", "infer"]:
            sample_ids_trn, sample_ids_val, sample_ids_tst = self.get_sample_partition(input_reader)
            print("Train: %d; Validation: %d; Test: %d" % (len(sample_ids_trn), len(sample_ids_val), len(sample_ids_tst)))
        elif self.opts.mode in ["infer_hero", "infer_conquer"]:
            sample_ids_trn, sample_ids_val = None, None
            sample_ids_tst = self.get_real_inference_sample(input_reader)
        else:
            assert False

        if self.opts.mode == "train":
            self.train_from_scratch(input_reader, sample_ids_trn, sample_ids_val, sample_ids_tst)
        elif self.opts.mode in ["infer", "infer_hero", "infer_conquer"]:
            self.infer(input_reader, sample_ids_tst)
        else:
            print("Error: Invalid mode '%s'" % self.opts.mode)
            raise NotImplementedError

    def train_from_scratch(self, input_reader, sample_ids_trn, sample_ids_val, sample_ids_tst):
        if self.opts.resume:
            self.load_model(self.best_model_path)
            training_range = range(self.checkpoint["epoch"] + 1, self.num_epoch)
            print(f"Resume training from Epoch {self.checkpoint['epoch'] + 1}.")
        else:
            training_range = range(self.num_epoch)

        if self.opts.vr_initialize:
            assert self.opts.task == "ivcml", "Error: cannot initialize model by pretrained vr in %s task" % self.opts.task
            self.load_vr_pretrain()

        for i_epoch in training_range:
            self.i_epoch = i_epoch
            self.train_epoch(input_reader, sample_ids_trn)
            self.env.summary()
            self.env.clear_cache()
            if i_epoch % self.evaluate_period == 0:
                metric_epoch = self.evaluate(input_reader, sample_ids_val, mode="val")
                self.env.summary()
                self.env.clear_cache()
                self.judge(metric_epoch)
            if self.is_early_stop:
                print("Early stop happens.")
                break
            if self.is_decay:
                self.scheduler.step()
            print("Learning rate @ E-%d:" % self.i_epoch, self.scheduler.get_last_lr())

            self.clear_cache()

        # Test model
        self.load_model(self.best_model_path)
        metric_test = self.evaluate(input_reader, sample_ids_tst, mode="test")
        self.env.summary()
        self.env.clear_cache()

    def infer(self, input_reader, sample_ids_tst):
        self.is_inference = True
        random_init = self.opts.random_policy
        if random_init:
            print("Randomly initialize without loading the trained model.")
        else:
            self.load_model(self.best_model_path)

        print("Evaluate on test:")
        self.evaluate(input_reader, sample_ids_tst, mode="test")
        self.clear_cache()

    def batch_generator(self, input_reader, sample_ids, is_train=True):
        data_size = len(sample_ids)
        batch_num = data_size // self.batch_size + int((data_size % self.batch_size) != 0)

        c_sample = 0
        with tqdm(range(batch_num), total=batch_num) as t:
            for i in t:
                st = i * self.batch_size
                ed = st + self.batch_size
                batch_sample_ids = sample_ids[st: ed]
                c_sample += len(batch_sample_ids)
                query_id_batch = []
                targets_batch, start_batch = [], []
                query_text_batch = []
                input_ids_bert_batch, input_mask_bert_batch, segment_ids_bert_batch = [], [], []
                proposal_batch, distance_proposal_batch = [], []
                bow_vec_batch = []
                tok_ids_batch = []
                query_rank_batch = []
                init_distance_batch = []
                for sample_id in batch_sample_ids:
                    query_id, start_id, node_ids, query_text, \
                        input_ids_bert, input_mask_bert, segment_ids_bert, \
                        bow_vec, tok_ids, query_rank, init_distance, \
                        proposal, distance_proposal = load_query_input(sample_id, input_reader, self.vocab)
                    query_text_batch.append(query_text)
                    input_ids_bert_batch.append(input_ids_bert)
                    input_mask_bert_batch.append(input_mask_bert)
                    segment_ids_bert_batch.append(segment_ids_bert)
                    query_id_batch.append(query_id)
                    start_batch.append(start_id)
                    targets_batch.append(node_ids)
                    bow_vec_batch.append(bow_vec)
                    tok_ids_batch.append(tok_ids)
                    query_rank_batch.append(query_rank)
                    init_distance_batch.append(init_distance)
                    proposal_batch.append(proposal)
                    distance_proposal_batch.append(distance_proposal)

                input_ids_bert_batch = pad_sequence(input_ids_bert_batch, batch_first=True)
                input_mask_bert_batch = pad_sequence(input_mask_bert_batch, batch_first=True)
                segment_ids_bert_batch = pad_sequence(segment_ids_bert_batch, batch_first=True)
                tok_ids_batch = pad_sequence(tok_ids_batch, batch_first=True)
                bow_vec_batch = torch.tensor(np.concatenate(bow_vec_batch, axis=0), dtype=torch.float32)

                targets_batch = pad_sequence(targets_batch, batch_first=True, padding_value=-2)  # default final pred is -1
                start_batch = torch.tensor(start_batch, dtype=torch.long)
                query_id_batch = torch.tensor(query_id_batch, dtype=torch.long)
                query_rank_batch = torch.tensor(query_rank_batch, dtype=torch.long)
                init_distance_batch = torch.tensor(init_distance_batch, dtype=torch.long)
                proposal_batch = torch.tensor(proposal_batch, dtype=torch.long)
                distance_proposal_batch = torch.tensor(distance_proposal_batch, dtype=torch.long)

                if is_train:
                    ce_loss, tri_loss, value_loss, pi_loss, pi_entropy = self.batch_loss_value()

                    info = f"E:{self.i_epoch}|"
                    info_terms = [(f"π:{pi_loss :.3f}|", pi_loss),
                                  (f"v:{value_loss :.3f}|", value_loss),
                                  (f"ce:{ce_loss :.3f}|", ce_loss),
                                  (f"tri:{tri_loss :.3f}|", tri_loss),
                                  (f"H:{pi_entropy :.3f}|", pi_entropy)]

                    for info_term, v in info_terms:
                        if v:
                            info += info_term

                    info += f"BA:{self.best_metrics['acc']:.3f}|" \
                            f"NIE:{self.no_improve_epoch_num}/{self.early_stop_epoch_num}|" \
                            f"mf:{self.env.c_missed_node}/{len(self.env.missed_feature_id)}"

                    t.set_description(desc=info)

                yield i, (input_ids_bert_batch, input_mask_bert_batch, segment_ids_bert_batch, tok_ids_batch, bow_vec_batch, start_batch, targets_batch, query_id_batch, query_rank_batch,
                          init_distance_batch, proposal_batch, distance_proposal_batch, query_text_batch)
        self.cur_data_size = 0
        print("Generated %d samples in total" % c_sample)

    def iterate(self, batch):
        input_ids_bert_batch, input_mask_bert_batch, segment_ids_bert_batch, tok_ids_batch, bow_vec_batch, \
            start_batch, targets_batch, query_id_batch, query_rank_batch, init_distance_batch, \
            proposal, distance_proposal, query_text_batch = cuda(batch, self.device)

        outputs = self.dqn(input_ids_bert_batch, input_mask_bert_batch, segment_ids_bert_batch, tok_ids_batch, bow_vec_batch, start_batch, query_id_batch, targets_batch)
        return (*outputs, query_rank_batch, init_distance_batch, start_batch, targets_batch)

    def train_epoch(self, input_reader, sample_ids_trn):
        self.dqn.train()
        self.dqn.training = True
        self.env.train_mode()

        for i_batch, batch_data in self.batch_generator(input_reader, sample_ids_trn, is_train=True):
            policy_pred_batch, value_pred_batch, last_value_batch, \
                policy_gold_batch, gae_batch, returns_batch, \
                policy_mask_batch, step_mask_batch, states_batch, concept_steps, \
                target_batch, query_ids, query_rank, init_distance, start_batch_verify, targets_batch_verify = self.iterate(batch_data)

            # index valid steps
            valid_step_mask = step_mask_batch[:, :-1].nonzero(as_tuple=True)  # step_mask_batch includes last state.
            policy_pred_batch = policy_pred_batch[valid_step_mask]
            value_pred_batch = value_pred_batch[valid_step_mask]
            gae_batch = gae_batch[valid_step_mask]
            returns_batch = returns_batch[valid_step_mask]
            policy_mask_batch = policy_mask_batch[valid_step_mask]
            policy_gold_batch = policy_gold_batch[valid_step_mask]

            if self.normalize_advantage:
                gae_batch = (gae_batch - gae_batch.mean()) / (gae_batch.std() + 1e-8)

            mask = policy_gold_batch >= 0
            self.c_covered_step_nk += mask.sum().item()
            self.c_out_step_nk += (policy_gold_batch < 0).sum().item()
            coo_covered = mask.nonzero(as_tuple=True)
            policy_pred_batch, policy_mask_batch, policy_gold_batch = policy_pred_batch[coo_covered], policy_mask_batch[coo_covered], policy_gold_batch[coo_covered]
            value_pred_batch, gae_batch, returns_batch = value_pred_batch[coo_covered], gae_batch[coo_covered], returns_batch[coo_covered]

            # back propagation
            loss = self.calc_loss(policy_pred_batch, policy_mask_batch, policy_gold_batch, value_pred_batch, gae_batch, returns_batch)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.max_grad_norm)
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, input_reader, query_ids_evaluate, mode="val"):
        self.dqn.eval()
        self.dqn.training = False
        self.env.eval_mode()

        policy_pred_eval, policy_gold_eval, value_pred_eval = [], [], []
        num_step_used = []
        for i_batch, batch_data in self.batch_generator(input_reader, query_ids_evaluate, is_train=True):
            policy_pred_batch, value_pred_batch, last_value_batch, \
                policy_gold_batch, gae_batch, returns_batch, \
                neighbor_mask_batch, step_mask_batch, states_batch, concept_steps, \
                target_batch, query_ids, query_rank, init_distance, start_batch_verify, targets_batch_verify = self.iterate(batch_data)

            # bad variable name: the initialized states at each step and,
            # meanwhile, they are also the predicted states in last time step.
            distance_batch = self.env.get_all_steps_distance_to_moment(states_batch.detach().cpu(), query_ids)

            # drop start to calculate number of steps used
            num_step_used_batch = step_mask_batch.sum(dim=-1)
            num_step_used.append(num_step_used_batch)

            self.record_and_analyze_inference(states_batch, concept_steps, target_batch, query_ids, query_rank, distance_batch, init_distance, num_step_used_batch)

            # drop the last to index the valid samples
            step_mask_init_batch = step_mask_batch[:, :-1]
            valid_step_mask = step_mask_init_batch.nonzero(as_tuple=True)
            policy_pred_batch = policy_pred_batch[valid_step_mask].detach().cpu()
            neighbor_mask_batch = neighbor_mask_batch[valid_step_mask].detach().cpu()
            policy_gold_batch = policy_gold_batch[valid_step_mask].detach().cpu()

            invalid_nei_coo = torch.logical_not(neighbor_mask_batch).nonzero(as_tuple=True)
            policy_pred_batch[invalid_nei_coo] = -1e5

            policy_pred_batch = policy_pred_batch.argmax(dim=-1)
            policy_gold_eval.append(policy_gold_batch)
            policy_pred_eval.append(policy_pred_batch)

        print("Overall Step-wise Accuracy:")
        print(self.opts.training_prefix, "\n****** %s ******" % self.opts.mode)
        metrics = self.calc_path_metric(num_step_used, mode)
        self.summary()
        return metrics

    def calc_metrics(self, gold_eval, pred_eval, mode):
        if pred_eval.sum():
            acc = accuracy_score(gold_eval, pred_eval)
            recall = recall_score(gold_eval, pred_eval)
            precision = precision_score(gold_eval, pred_eval)
            f1 = f1_score(gold_eval, pred_eval)
        else:
            acc, recall, precision, f1 = -1, -1, -1, -1

        metrics = {
            "acc": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }

        if mode == "val":
            res_info = f"Acc: {acc:.6f} | " \
                       f"Best Acc: {max(acc, self.best_metrics['acc']):.6f} | " \
                       f"NIE: {self.no_improve_epoch_num}/{self.early_stop_epoch_num}"
        else:
            res_info = f"Acc: {acc:.6f}"

        info = {
            "val": "Epoch: %d Validation %s " % (self.i_epoch, res_info),
            "test": "Test %s" % res_info
        }[mode]
        print(info)

        return metrics

    def calc_step_metric(self, gold_list, pred_list, mode):
        print("Step Accuracy:", end=" ")
        pred = torch.cat(gold_list, dim=0)
        gold = torch.cat(pred_list, dim=0)
        pred = (pred == gold).to(torch.long).detach().cpu()
        gold = torch.ones_like(pred, dtype=torch.long, device=torch.device("cpu"))
        metrics = self.calc_metrics(gold, pred, mode)
        return metrics

    def calc_path_metric(self, num_step_used, mode):
        num_step_used = torch.cat(num_step_used, dim=0)
        pred = (num_step_used <= self.infer_steps).to(torch.long).detach().cpu()
        gold = torch.ones_like(pred, dtype=torch.long, device=torch.device("cpu"))
        n_true = pred.sum().item()
        n_false = (1 - pred).sum().item()
        ttl = n_true + n_false
        print("Path Accuracy: %d/%d | False: %d/%d |" % (n_true, ttl, n_false, ttl), end=" ")
        metrics = self.calc_metrics(gold, pred, mode)
        return metrics

    def judge(self, metrics):
        best_metrics_upd = {}
        for key in self.metric_cats:
            best_metrics_upd[key] = max(metrics[key], self.best_metrics[key])
            if key != self.judge_metric:
                continue
            if metrics[key] > self.best_metrics[key] + 1e-5:
                self.no_improve_epoch_num = 0
                self.save_model(metrics)
            else:
                self.no_improve_epoch_num += 1

            if self.no_improve_epoch_num >= 5 and self.no_decay_epoch_num >= 5:
                self.is_decay = True
                self.no_decay_epoch_num = 0
            else:
                self.no_decay_epoch_num += 1
                self.is_decay = False

        early_stop = self.no_improve_epoch_num == self.early_stop_epoch_num
        self.best_metrics = best_metrics_upd
        self.is_early_stop = early_stop

    def save_model(self, metrics):
        ce_loss, tri_loss, value_loss, policy_loss, entropy_loss = self.epoch_loss_value()

        checkpoint = {
            'epoch': self.i_epoch,
            'model_state_dict': self.dqn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ce_loss': ce_loss,
            'tri_loss': tri_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
        }
        checkpoint.update(metrics)
        torch.save(checkpoint, self.best_model_path)
        print("Model '%s' saved!" % self.best_model_path)

        del checkpoint["model_state_dict"]
        del checkpoint["optimizer_state_dict"]
        pprint(checkpoint)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, self.device)
        self.dqn.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint["model_state_dict"]
        del checkpoint["optimizer_state_dict"]
        print("Model '%s' loaded!" % model_path)
        self.checkpoint = checkpoint
        pprint(checkpoint)

    def load_vr_pretrain(self):
        if self.opts.dataset == "didemo":
            model_path = os.path.join(self.opts.ckpt_root, f"{self.opts.dataset}_vr_fbert_wms_gru_best_r_match")
        elif self.opts.dataset == "tvr":
            model_path = os.path.join(self.opts.ckpt_root, f"{self.opts.dataset}_vr_fbert_gru_best_r_match")
        else:
            print("Unsupported dataset %s" % self.opts.dataset)
            raise NotImplementedError

        vr_checkpoint = torch.load(model_path, self.device)
        vr_param_keys = vr_checkpoint["model_state_dict"].keys()
        param_keys = self.dqn.state_dict().keys()
        shared_keys = [k for k in param_keys if k in vr_param_keys]
        state_dict = OrderedDict([(k, vr_checkpoint["model_state_dict"][k] if k in vr_param_keys else self.dqn.state_dict()[k]) for k in param_keys])
        self.dqn.load_state_dict(state_dict)
        print("VR Model '%s' loaded!" % model_path)
        print("The following parameters (%d) are loaded:" % len(shared_keys))
        pprint(sorted(shared_keys, reverse=True))
        del vr_checkpoint["model_state_dict"]
        del vr_checkpoint["optimizer_state_dict"]
        pprint(vr_checkpoint)

    def clear_cache(self):
        self.c_covered_step_nk = 0
        self.c_out_step_nk = 0

        self.step_used = {}
        self.traj_collection = {}
        self.traj_degree = []
        self.c_recorded_sample = 0

        self.ce_loss_epoch = []
        self.policy_loss_epoch = []
        self.value_loss_epoch = []
        self.entropy_loss_epoch = []

    # loss
    def calc_loss(self, policy_pred_batch, policy_mask_batch, policy_gold_batch, value_pred_batch, gae_batch, returns_batch):
        loss = []
        if "a2c" in self.loss_func:
            a2c_weight = self.loss_weight[self.loss_func.index("a2c")]
            a2c_loss = self.calc_a2c_loss(policy_pred_batch, value_pred_batch, gae_batch, returns_batch, a2c_weight)
            loss.append(a2c_loss)

        if "ce" in self.loss_func:
            ce_weight = self.loss_weight[self.loss_func.index("ce")]
            ce_loss = ce_weight * self.ce_loss_func(policy_pred_batch, policy_gold_batch)
            self.ce_loss_epoch.append(ce_loss.item())
            loss.append(ce_loss)

        if "triplet" in self.loss_func:
            triplet_weight = self.loss_weight[self.loss_func.index("triplet")]
            tri_loss = triplet_weight * self.triplet_loss_func.ranking_loss_one_to_more(policy_pred_batch, policy_gold_batch, policy_mask_batch)
            self.triplet_loss_epoch.append(tri_loss.item())
            loss.append(tri_loss)

        loss = sum(loss)
        return loss

    def calc_a2c_loss(self, policy_pred_batch, value_pred_batch, advantage, returns, weight=1.0):
        # Gradient Loss
        pi = F.softmax(policy_pred_batch, dim=-1).max(dim=-1, keepdim=False).values
        neg_log_prob = - torch.log(pi)
        policy_entropy = - torch.mean(neg_log_prob * pi)
        policy_loss = (neg_log_prob * advantage).mean()
        # Value Loss
        value_loss = self.mse_loss_func(value_pred_batch, returns)
        a2c_loss = policy_loss + self.ent_coef * policy_entropy + self.vf_coef * value_loss

        self.policy_loss_epoch.append(weight*policy_loss.item())
        self.value_loss_epoch.append(weight*self.vf_coef*value_loss.item())
        self.entropy_loss_epoch.append(weight*self.ent_coef*policy_entropy.item())
        return weight*a2c_loss

    def load_input_reader(self):
        tag_graph_param = f"wc_{self.opts.weight_concept:.2f}_ws_{self.opts.weight_subtitle:.2f}_ths_{self.opts.edge_threshold_single:.2f}_thx_{self.opts.edge_threshold_cross:.2f}"
        if self.opts.use_balanced_graph:
            tag_graph_param += f"_top_{self.opts.k_ei}"
        if self.opts.mode in ["train", "infer"]:
            input_tag = f"clip_level_input_{tag_graph_param}"
        elif self.opts.mode in ["infer_hero"]:
            input_tag = f"clip_level_input_hero_{tag_graph_param}"
        elif self.opts.mode in ["infer_conquer"]:
            input_tag = f"clip_level_input_conquer_{tag_graph_param}"
        else:
            assert False

        input_reader = LMDBFile(self.opts.data_root, self.feature_type, self.opts.split, tag=input_tag, encode_method="json", readonly=True)
        return input_reader

    def get_sample_partition(self, input_reader):
        """ Sample partition for training, validation and testing """

        # parsing from sample id
        def __sample_id_to_distance(s_id):
            return int(s_id.split("_")[2])

        def __sample_id_to_query_id(s_id):
            return s_id.split("_")[0]

        def __sample_id_to_split(s_id):
            return s_id.split("_")[3]

        sample_ids = input_reader.keys()
        sample_id_split = {"train": [], "val": [], "test": []}
        split_c_legal_d = {"train": 0, "val": 0, "test": 0}
        split_c_legal_n = {"train": 0, "val": 0, "test": 0}
        cc_size_minimum = 600

        # prepare connected component (cc) size for filtering
        node_cc_size = [-1 for _ in self.env.graph.nodes()]
        ccs = nx.algorithms.components.connected_components(self.env.graph)
        for cc in ccs:
            cc_size = len(cc)
            for n in cc:
                node_cc_size[n] = cc_size

        # process training samples
        for sample_id in tqdm(sample_ids, total=len(sample_ids), desc="Splitting..."):
            split = __sample_id_to_split(sample_id)

            # 1. filter by initial distance
            is_legal_d = self.minimum_distance <= __sample_id_to_distance(sample_id) <= self.maximum_distance
            if not is_legal_d:
                continue
            split_c_legal_d[split] += 1

            # 2. filter by connected component size
            tgt_probe = input_reader[sample_id]["target"][0]
            cc_size = node_cc_size[tgt_probe]
            if cc_size < cc_size_minimum:
                continue
            split_c_legal_n[split] += 1

            # split valid samples
            sample_id_split[split].append(sample_id)

        for split in split_c_legal_d:
            print("%s: Get %d/%d with cc size > %d." % (split, split_c_legal_n[split], split_c_legal_d[split], cc_size_minimum))

        if sample_id_split["test"]:
            return sample_id_split["train"], sample_id_split["val"], sample_id_split["test"]

        # Split training set into train and validation in case of no testing set
        id_by_tgt = {}
        for sample_id in sample_id_split["train"]:
            tgt = __sample_id_to_query_id(sample_id)
            if tgt not in id_by_tgt:
                id_by_tgt[tgt] = []
            id_by_tgt[tgt].append(sample_id)

        target_ids = list(id_by_tgt.keys())
        random.seed(42)
        random.shuffle(target_ids)

        query_size_ttl = len(target_ids)
        boundary = int(query_size_ttl * 0.8)
        target_ids_trn = target_ids[:boundary]
        target_ids_val = target_ids[boundary:]

        sample_ids_trn = flatten([id_by_tgt[tgt] for tgt in target_ids_trn])
        sample_ids_val = flatten([id_by_tgt[tgt] for tgt in target_ids_val])
        sample_ids_test = sample_id_split["val"]
        return sample_ids_trn, sample_ids_val, sample_ids_test

    def get_real_inference_sample(self, input_reader):
        sample_ids = input_reader.keys()
        sample_ids_tst = []
        inference_split = "val" if self.opts.dataset == "tvr" else "test"
        for sample_id in tqdm(sample_ids, total=len(sample_ids), desc="Splitting..."):
            if input_reader[sample_id]["split"] != inference_split:
                continue
            if int(input_reader[sample_id]["distance"]) < MAX_MOMENT_DISTANCE:  # update distance
                distance = int(input_reader[sample_id]["distance"])
                self.maximum_distance = max(self.maximum_distance, distance)
                self.minimum_distance = min(self.minimum_distance, distance)
            sample_ids_tst.append(sample_id)
        return sample_ids_tst

    # display functions
    def epoch_loss_value(self):
        ce_loss = mean(self.ce_loss_epoch) if self.ce_loss_epoch else 0
        tri_loss = mean(self.triplet_loss_epoch) if self.triplet_loss_epoch else 0
        value_loss = mean(self.value_loss_epoch) if self.value_loss_epoch else 0
        policy_loss = mean(self.policy_loss_epoch) if self.policy_loss_epoch else 0
        entropy_loss = mean(self.entropy_loss_epoch) if self.entropy_loss_epoch else 0
        return ce_loss, tri_loss, value_loss, policy_loss, entropy_loss

    def batch_loss_value(self):
        ce_loss = self.ce_loss_epoch[-1] if self.ce_loss_epoch else 0
        tri_loss = self.triplet_loss_epoch[-1] if self.triplet_loss_epoch else 0
        value_loss = self.value_loss_epoch[-1] if self.value_loss_epoch else 0
        policy_loss = self.policy_loss_epoch[-1] if self.policy_loss_epoch else 0
        policy_entropy = self.entropy_loss_epoch[-1] if self.entropy_loss_epoch else 0
        return ce_loss, tri_loss, value_loss, policy_loss, policy_entropy

    def record_and_analyze_inference(self, states_with_last: torch.Tensor, concept_steps: torch.Tensor,
                                     target_ids: torch.Tensor, query_ids: torch.Tensor, query_rank: torch.Tensor,
                                     state_distance: torch.Tensor, init_distance: torch.Tensor, step_used: torch.Tensor):

        batch_size = states_with_last.shape[0]
        initial_distance = state_distance[:, 0].tolist()
        target_ids = target_ids.detach().cpu()

        for i_sample in range(batch_size):
            init_d = initial_distance[i_sample]
            tgt_ids = target_ids[i_sample].tolist()
            query_id = query_ids[i_sample].item()
            rank = query_rank[i_sample].item()
            if init_d not in self.step_used:
                self.step_used[init_d] = {}

            num_step = step_used[i_sample].item()
            if num_step not in self.step_used[init_d]:
                self.step_used[init_d][num_step] = 0

            self.step_used[init_d][num_step] += 1

            states_tra = states_with_last[i_sample][:num_step + 1].tolist()
            distances_tra = state_distance[i_sample][:num_step + 1].tolist()
            if not self.opts.no_feedback and not self.opts.random_policy:
                concepts = concept_steps[i_sample][:num_step + 1].tolist()
            else:
                concepts = []

            key = "d%ds%d" % (init_d, num_step)
            if key not in self.traj_collection:
                self.traj_collection[key] = {}
            query_key = "%d_%d" % (query_id, distances_tra[0])
            assert query_key not in self.traj_collection[key], "Error, exists %s." % query_key
            self.traj_collection[key][query_key] = {
                "state": states_tra,
                "distance": distances_tra,
                "target": tgt_ids,
                "concepts": concepts,
                "rank": rank
            }
            self.c_recorded_sample += 1

    def summary(self):
        print("————————————————————————————————————————————————————————————————————————————————")
        print("Max distance:", self.maximum_distance)

        max_step = self.infer_steps + 3  # one for start, one for failed, one for check
        c_sample_per_distance = {i: 0 for i in range(0, self.maximum_distance + 3)}
        c_sample_per_distance[int(MAX_MOMENT_DISTANCE)] = 0
        distances = list(range(self.minimum_distance, self.maximum_distance + 1))
        if self.opts.mode in ["infer_hero", "infer_conquer"]:
            distances += [int(MAX_MOMENT_DISTANCE)]

        c_ttl = 0
        print("Distance:", self.step_used.keys())
        print("#Step     " + " ".join(["%6d" % i for i in range(0, max_step)]))
        for init_d in distances:
            print("D=%-6d" % init_d, end="  |")
            for num_step_used in range(0, max_step):
                step_c = self.step_used[init_d][num_step_used] if ((init_d in self.step_used) and (num_step_used in self.step_used[init_d])) else 0
                c_sample_per_distance[init_d] += step_c
                print("%4d" % step_c, end=" ")
            print("(%4d)" % c_sample_per_distance[init_d])
            c_ttl += c_sample_per_distance[init_d]
        print("%d samples in total." % c_ttl)
        print()

        print("#Step     " + " ".join(["%6d" % i for i in range(0, max_step)]))
        for init_d in distances:
            print("D=%-6d" % init_d, end="  |")
            for num_step_used in range(0, max_step):
                step_c = self.step_used[init_d][num_step_used] if ((init_d in self.step_used) and (num_step_used in self.step_used[init_d])) else 0
                rate = step_c / c_sample_per_distance[init_d] if c_sample_per_distance[init_d] else 0
                print("%.4f" % rate, end=" ")
            print()
        print()

        ttl = self.c_out_step_nk + self.c_covered_step_nk
        print("Recorded %d samples in total" % self.c_recorded_sample)
        if ttl:
            print("Covered step: %d/%d (%.2f), Out: %d/%d (%.2f)" % (self.c_covered_step_nk, ttl, self.c_covered_step_nk / ttl, self.c_out_step_nk, ttl, self.c_out_step_nk / ttl))

        if self.is_inference:
            action_tag = "N%d" % self.opts.k_for_nk if self.opts.action_type == "nk" else "N1"
            f_path = 'traj/traj_%s_%s_%s.json' % (self.opts.mode, action_tag, self.training_prefix)
            if not os.path.exists("./traj"):
                os.makedirs("./traj")
            with open(f_path, "w") as f:
                json.dump(self.traj_collection, f)
            print('%s saved.' % f_path)

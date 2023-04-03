from pprint import pprint
import numpy as np
import torch
from torch import optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from data import LMDBFile
from load import load_query_input_vr, load_concepts
from loss import TripletLoss, FavorPositiveBCELoss, calc_similarity
from model_vr import DeepVR
from util import cuda, mean
from vocab import Vocab
from evaluation import calc_rank, mean_average_precision, evaluate_rank


class TrainerVR(object):
    def __init__(self, opts):
        super(TrainerVR, self).__init__()
        self.opts = opts

        # Setting
        self.is_inference = True
        self.batch_size = opts.batch_size
        # # Training Setting
        self.num_epoch = 500
        self.evaluate_period = 1
        self.early_stop_epoch_num = opts.early_stop_epoch_num
        self.feature_type = opts.feature_type
        self.training_prefix = opts.training_prefix
        self.judge_metric = "r_match"
        self.metric_cats = ["r_match", "r_cls", "r_fuse", "r_fuse_map"]
        self.best_model_path = "%s/ckpt/%s_best_%s" % (opts.data_root, self.training_prefix, self.judge_metric)
        print("Best checkpoint will be stored at: %s." % self.best_model_path)

        # Session Record
        self.i_epoch = 0
        self.no_improve_epoch_num = 0
        self.is_early_stop = False
        self.best_metrics = {"r_match": 0, "r_cls": 0, "r_fuse": 0, "r_fuse_map": 0}

        # # epoch cache
        self.cur_data_size = 0
        self.ce_loss_epoch = []
        self.cls_loss_epoch = []

        # Input Reader
        tag_graph_param = "wc_%.2f_ws_%.2f_ths_%.2f_thx_%.2f" % (self.opts.weight_concept, self.opts.weight_subtitle, self.opts.edge_threshold_single, self.opts.edge_threshold_cross)
        if opts.use_balanced_graph:
            tag_graph_param += "_top_%d" % self.opts.k_ei
        tag_moment_feat = "clip_level_moment_feat_%s" % tag_graph_param
        self.moment_feat_reader = LMDBFile(opts.data_root, self.feature_type, opts.split, tag=tag_moment_feat, encode_method="ndarray", readonly=True)
        tag_vr_input_data = "clip_level_vr_input_%s" % tag_graph_param
        self.input_reader = LMDBFile(opts.data_root, self.feature_type, opts.split, tag=tag_vr_input_data, encode_method="json", readonly=True)
        concepts, concept_size = load_concepts(opts.concept_list_path)
        self.vocab = Vocab(concepts, opts.stem)

        # Model
        self.device = torch.device("cuda")
        self.model = DeepVR(768, self.vocab.concept_size_en, self.vocab.concept_size_de, self.opts)
        self.model.to(self.device)
        self.grad_clip = 2
        self.optimizer = optim.Adam(self.model.parameters(), lr=opts.init_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Loss
        self.ce_loss_func = torch.nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=self.opts.margin,
                                        measure="cosine",
                                        max_violation=False,
                                        cost_style="all",
                                        direction="all",
                                        seq_match=False,
                                        pos_loss_term=False)
        self.cls_loss_func = FavorPositiveBCELoss()

    def run(self):
        sample_ids_trn, sample_ids_val, sample_ids_tst = self.get_sample_partition()
        print("Train: %d; Validation: %d; Test: %d" % (len(sample_ids_trn), len(sample_ids_val), len(sample_ids_tst)))
        if self.opts.mode == "train":
            self.train_from_scratch(sample_ids_trn, sample_ids_val, sample_ids_tst)
        elif self.opts.mode == "infer":
            self.infer(sample_ids_tst)
        else:
            print("Error: Invalid mode '%s'" % self.opts.mode)
            raise NotImplementedError

    def train_from_scratch(self, sample_ids_trn, sample_ids_val, sample_ids_tst):
        if self.opts.resume:
            self.load_model(self.best_model_path)
            training_range = range(self.checkpoint["epoch"]+1, self.num_epoch)
            print(f"Resume training from Epoch {self.checkpoint['epoch']+1}.")
        else:
            training_range = range(self.num_epoch)

        for i_epoch in training_range:
            self.i_epoch = i_epoch
            self.train_epoch(sample_ids_trn)
            if i_epoch % self.evaluate_period == 0:
                metric_epoch = self.evaluate(sample_ids_val, mode="val")
                self.judge(metric_epoch)
            if self.is_early_stop:
                print("Early stop happens.")
                break
            self.scheduler.step()
            self.clear_epoch_cache()

        # Test model
        self.load_model(self.best_model_path)
        metric_test = self.evaluate(sample_ids_tst, mode="test")

    def infer(self, sample_ids_tst):
        self.is_inference = True
        random_init = self.opts.random_policy
        if random_init:
            print("Randomly initialize without loading the trained model.")
        else:
            self.load_model(self.best_model_path)
        print("Evaluate on test:")
        self.infer_all(sample_ids_tst, mode="test")

    def batch_generator(self, sample_ids, is_train=True):
        data_size = len(sample_ids)
        batch_num = data_size // self.batch_size + int((data_size % self.batch_size) != 0)

        c_sample = 0
        c_bad_query = 0
        with tqdm(range(batch_num), total=batch_num) as t:
            for i in t:
                st = i * self.batch_size
                ed = st + self.batch_size
                batch_sample_ids = sample_ids[st: ed]
                c_sample += len(batch_sample_ids)
                text_bert_ids_batch = []
                query_feat_batch = []
                concept_gold_batch = []
                tok_ids_batch = []
                mask_bert_batch = []
                segment_ids_batch = []

                for sample_id in batch_sample_ids:
                    text_bert_id, mask_bert, segment_ids, query_feat, concept_gold, tok_ids = load_query_input_vr(sample_id, self.input_reader, self.moment_feat_reader, self.vocab)
                    if concept_gold.sum() == 0:
                        c_bad_query += 1
                        continue
                    text_bert_ids_batch.append(text_bert_id)
                    mask_bert_batch.append(mask_bert)
                    segment_ids_batch.append(segment_ids)
                    query_feat_batch.append(query_feat)
                    tok_ids_batch.append(tok_ids)
                    concept_gold_batch.append(concept_gold)

                text_bert_ids_batch = pad_sequence(text_bert_ids_batch, batch_first=True, padding_value=0)
                mask_bert_batch = pad_sequence(mask_bert_batch, batch_first=True, padding_value=0)
                segment_ids_batch = pad_sequence(segment_ids_batch, batch_first=True, padding_value=0)
                tok_ids_batch = pad_sequence(tok_ids_batch, batch_first=True, padding_value=0)
                query_feat_batch = torch.stack(query_feat_batch, dim=0)
                concept_gold_batch = torch.tensor(np.concatenate(concept_gold_batch, axis=0), dtype=torch.float32)

                if is_train:
                    ce_loss, cls_loss = self.batch_loss_value()

                    info = f"E:{self.i_epoch}| #BQ:{c_bad_query}|"
                    info_terms = [(f"ce:{ce_loss :.3f}|", ce_loss),
                                  (f"cls:{cls_loss :.3f}|", cls_loss)]

                    for info_term, value in info_terms:
                        if value:
                            info += info_term

                    info += f"BS:{self.best_metrics[self.judge_metric]:.3f} |" \
                            f"NIE:{self.no_improve_epoch_num}/{self.early_stop_epoch_num}"

                    t.set_description(desc=info)

                yield i, (text_bert_ids_batch, mask_bert_batch, segment_ids_batch, query_feat_batch, concept_gold_batch, tok_ids_batch)
        self.cur_data_size = 0
        print("Generated %d samples in total" % c_sample)

    def train_epoch(self, sample_ids_trn):
        self.model.train()
        self.model.training = True

        for i_batch, batch_data in self.batch_generator(sample_ids_trn, is_train=True):
            moment_feats_batch, query_feat_batch, concept_pred_batch, concept_gold_batch = self.iterate(batch_data)
            match_loss = self.triplet_loss(query_feat_batch, moment_feats_batch)

            cls_loss = self.cls_loss_func(concept_pred_batch, concept_gold_batch)
            loss = cls_loss + match_loss
            # loss = match_loss
            self.cls_loss_epoch.append(cls_loss.item())
            self.ce_loss_epoch.append(match_loss.item())

            self.optimizer.zero_grad()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            loss.backward()
            self.optimizer.step()

        e_ce_loss, e_cls_loss = self.epoch_loss_value()

    def evaluate(self, query_ids_evaluate, mode="val"):
        self.model.eval()
        self.model.training = False
        n_caption = 1

        cap_size, vid_size, mean_ap_c2v, mean_ap_v2c = 0, 0, 0, 0
        policy_pred_eval, policy_gold_eval, value_pred_eval = [], [], []
        ranks_c2v_match, ranks_v2c_match, ranks_c2v, ranks_v2c, ranks_c2v_cls, ranks_v2c_cls = [], [], [], [], [], []
        for i_batch, batch_data in self.batch_generator(query_ids_evaluate, is_train=True):
            moment_feats_batch, query_feat_batch, concept_pred_batch, concept_gold_batch = self.iterate(batch_data)

            match_c2v_sim = calc_similarity(moment_feats_batch, query_feat_batch, "cosine")
            concept_c2v_sim = calc_similarity(concept_pred_batch, concept_gold_batch, "cosine")

            # retrieval
            ranks_c2v_match_batch = calc_rank(match_c2v_sim, n_caption, eval_direction="c2i").cpu().detach().numpy()
            ranks_v2c_match_batch = calc_rank(match_c2v_sim, n_caption, eval_direction="i2c").cpu().detach().numpy()
            ranks_c2v_cls_batch = calc_rank(concept_c2v_sim, n_caption, eval_direction="c2i").cpu().detach().numpy()
            ranks_v2c_cls_batch = calc_rank(concept_c2v_sim, n_caption, eval_direction="i2c").cpu().detach().numpy()
            c2v_sim = match_c2v_sim * 0.7 + concept_c2v_sim * 0.3
            ranks_c2v_batch = calc_rank(c2v_sim, n_caption, eval_direction="c2i").cpu().detach().numpy()
            ranks_v2c_batch = calc_rank(c2v_sim, n_caption, eval_direction="i2c").cpu().detach().numpy()

            cap_size += query_feat_batch.shape[0]
            vid_size += moment_feats_batch.shape[0]
            mean_ap_c2v_batch = mean_average_precision(c2v_sim, n_caption, eval_direction="c2i")
            mean_ap_v2c_batch = mean_average_precision(c2v_sim, n_caption, eval_direction="i2c")
            mean_ap_c2v += mean_ap_c2v_batch
            mean_ap_v2c += mean_ap_v2c_batch

            ranks_c2v_match.append(ranks_c2v_match_batch)
            ranks_v2c_match.append(ranks_v2c_match_batch)
            ranks_c2v_cls.append(ranks_c2v_cls_batch)
            ranks_v2c_cls.append(ranks_v2c_cls_batch)
            ranks_c2v.append(ranks_c2v_batch)
            ranks_v2c.append(ranks_v2c_batch)

        # retrieval score
        ranks_c2v_match = np.concatenate(ranks_c2v_match, axis=0)
        ranks_v2c_match = np.concatenate(ranks_v2c_match, axis=0)
        print("match", end=" ")
        score_c2v_match = evaluate_rank(ranks_c2v_match, "c2i", epoch_num=self.i_epoch)
        score_v2c_match = evaluate_rank(ranks_v2c_match, "i2c", epoch_num=self.i_epoch)

        ranks_c2v_cls = np.concatenate(ranks_c2v_cls, axis=0)
        ranks_v2c_cls = np.concatenate(ranks_v2c_cls, axis=0)
        print("classification", end=" ")
        score_c2v_cls = evaluate_rank(ranks_c2v_cls, "c2i", epoch_num=self.i_epoch)
        score_v2c_cls = evaluate_rank(ranks_v2c_cls, "i2c", epoch_num=self.i_epoch)

        ranks_c2v = np.concatenate(ranks_c2v, axis=0)
        ranks_v2c = np.concatenate(ranks_v2c, axis=0)
        print("combine", end=" ")
        score_c2v = evaluate_rank(ranks_c2v, "c2i", epoch_num=self.i_epoch)
        score_v2c = evaluate_rank(ranks_v2c, "i2c", epoch_num=self.i_epoch)

        mean_ap_c2v = mean_ap_c2v / cap_size
        mean_ap_v2c = mean_ap_v2c / vid_size
        retrieval_score_match = (score_c2v_match + score_v2c_match) / 2
        retrieval_score_cls = (score_c2v_cls + score_v2c_cls) / 2
        retrieval_score = (score_c2v + score_v2c) / 2
        retrieval_map = (mean_ap_c2v + mean_ap_v2c) / 2
        metrics = {
            "r_match": retrieval_score_match,
            "r_cls": retrieval_score_cls,
            "r_fuse": retrieval_score,
            "r_fuse_map": retrieval_map
        }

        return metrics

    # def infer_all(self, query_ids_test, mode="test"):
    #     self.model.eval()
    #     self.model.training = False
    #     n_caption = 1
    #     cap_size, vid_size, mean_ap_c2v, mean_ap_v2c = 0, 0, 0, 0
    #     ranks_c2v_match, ranks_v2c_match, ranks_c2v, ranks_v2c, ranks_c2v_cls, ranks_v2c_cls = [], [], [], [], [], []
    #     for i_batch, batch_data in self.batch_generator(query_ids_test, is_train=True):
    #         text_ids_batch_o, query_feat_batch_o, concept_gold_batch_o = cuda(batch_data, self.device)
    #
    #         query_feat_all, concept_pred_all = [], []
    #         for i_batch, batch_data in self.batch_generator(query_ids_test, is_train=True):
    #             text_ids_batch_i, query_feat_batch_i, concept_gold_batch_i = cuda(batch_data, self.device)
    #             moment_feats_batch, query_feat_batch, concept_pred_batch = self.model(text_ids_batch_o, text_ids_batch_i)
    #             # moment_feats_batch = self.model.moment_encode(query_feat_batch)
    #             # match_c2v_sim = calc_similarity(moment_feats_batch, query_feat_all, "cosine")
    #             # concept_c2v_sim = calc_similarity(concept_pred_all, concept_gold_batch, "cosine")
    #
    #         # retrieval
    #         ranks_c2v_match_batch = calc_rank(match_c2v_sim, n_caption, eval_direction="c2i").cpu().detach().numpy()
    #         ranks_v2c_match_batch = calc_rank(match_c2v_sim, n_caption, eval_direction="i2c").cpu().detach().numpy()
    #         ranks_c2v_cls_batch = calc_rank(concept_c2v_sim, n_caption, eval_direction="c2i").cpu().detach().numpy()
    #         ranks_v2c_cls_batch = calc_rank(concept_c2v_sim, n_caption, eval_direction="i2c").cpu().detach().numpy()
    #         c2v_sim = match_c2v_sim * 0.7 + concept_c2v_sim * 0.3
    #         ranks_c2v_batch = calc_rank(c2v_sim, n_caption, eval_direction="c2i").cpu().detach().numpy()
    #         ranks_v2c_batch = calc_rank(c2v_sim, n_caption, eval_direction="i2c").cpu().detach().numpy()
    #
    #         cap_size += query_feat_batch.shape[0]
    #         vid_size += moment_feats_batch.shape[0]
    #         mean_ap_c2v_batch = mean_average_precision(c2v_sim, n_caption, eval_direction="c2i")
    #         mean_ap_v2c_batch = mean_average_precision(c2v_sim, n_caption, eval_direction="i2c")
    #         mean_ap_c2v += mean_ap_c2v_batch
    #         mean_ap_v2c += mean_ap_v2c_batch
    #
    #         ranks_c2v_match.append(ranks_c2v_match_batch)
    #         ranks_v2c_match.append(ranks_v2c_match_batch)
    #         ranks_c2v_cls.append(ranks_c2v_cls_batch)
    #         ranks_v2c_cls.append(ranks_v2c_cls_batch)
    #         ranks_c2v.append(ranks_c2v_batch)
    #         ranks_v2c.append(ranks_v2c_batch)

    def iterate(self, batch_data):
        text_ids_batch, mask_bert_batch, segment_ids_batch, query_feat_batch, concept_gold_batch, tok_ids_batch = cuda(batch_data, self.device)
        moment_feats_batch, query_feat_batch, concept_pred_batch = self.model(text_ids_batch, mask_bert_batch, segment_ids_batch, tok_ids_batch, concept_gold_batch, query_feat_batch)
        return moment_feats_batch, query_feat_batch, concept_pred_batch, concept_gold_batch

    def get_sample_partition(self):
        sample_ids = self.input_reader.keys()
        sample_id_split = {"train": [], "val": [], "test": []}

        # process training samples
        for query_id in tqdm(sample_ids, total=len(sample_ids), desc="Splitting..."):
            split = self.input_reader[query_id]["split"]
            sample_id_split[split].append(query_id)

        if sample_id_split["test"]:
            return sample_id_split["train"], sample_id_split["val"], sample_id_split["test"]

        query_size_ttl = len(sample_id_split["train"])
        boundary = int(query_size_ttl * 0.8)
        sample_ids_trn = sample_id_split["train"][:boundary]
        sample_ids_val = sample_id_split["train"][boundary:]
        sample_ids_test = sample_id_split["val"]
        return sample_ids_trn, sample_ids_val, sample_ids_test

    def clear_epoch_cache(self):
        self.cur_data_size = 0
        self.ce_loss_epoch = []
        self.cls_loss_epoch = []

    def summary(self):
        print("————————————————————————————————————————————————————————————————————————————————")
        pprint(self.best_metrics)
        print("————————————————————————————————————————————————————————————————————————————————")

    # information display functions
    def epoch_loss_value(self):
        ce_loss = mean(self.ce_loss_epoch) if self.ce_loss_epoch else 0
        cls_loss = mean(self.cls_loss_epoch) if self.cls_loss_epoch else 0
        return ce_loss, cls_loss

    def batch_loss_value(self):
        ce_loss = self.ce_loss_epoch[-1] if self.ce_loss_epoch else 0
        cls_loss = self.cls_loss_epoch[-1] if self.cls_loss_epoch else 0
        return ce_loss, cls_loss

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

        early_stop = self.no_improve_epoch_num == self.early_stop_epoch_num
        self.best_metrics = best_metrics_upd
        self.is_early_stop = early_stop

    def save_model(self, metrics):
        ce_loss, cls_loss = self.epoch_loss_value()

        checkpoint = {
            'epoch': self.i_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ce_loss': ce_loss,
        }
        checkpoint.update(metrics)
        torch.save(checkpoint, self.best_model_path)

        print("Model '%s' saved!" % self.best_model_path)

        del checkpoint["model_state_dict"]
        del checkpoint["optimizer_state_dict"]
        pprint(checkpoint)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for key in self.best_metrics:
            self.best_metrics[key] = checkpoint[key]

        del checkpoint["model_state_dict"]
        del checkpoint["optimizer_state_dict"]
        print("Model '%s' loaded!" % model_path)
        self.checkpoint = checkpoint
        pprint(checkpoint)

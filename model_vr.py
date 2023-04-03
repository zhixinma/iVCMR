import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, kernel_num, emb_dim, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, kernel_num, (window_size, emb_dim), padding=(window_size - 1, 0))
            for window_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        con_out = x.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)
        return con_out


class RetrievalBaseModel(nn.Module):
    def __init__(self, opts, d_fea, d_concept_en, d_concept_de):
        super(RetrievalBaseModel, self).__init__()
        # Configuration
        self.d_fea = d_fea
        self.d_concept_en = d_concept_en
        self.d_concept_de = d_concept_de
        self.text_mode = opts.text_mode.split(",")
        for mode in self.text_mode:
            assert mode in ["bert", "bow", "gru"]

        # Visual Encoder
        self.visual_feat_mlp = nn.Linear(d_fea, d_fea)

        # Textual Encoder
        self.num_feat = 0
        if "bert" in self.text_mode:
            self.num_feat += 1
            self.freeze_bert = opts.freeze_bert
            self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
            for p in self.bert_encoder.parameters():
                p.requires_grad = not self.freeze_bert
        if "bow" in self.text_mode:
            self.num_feat += 1
            self.bow_encoder = nn.Linear(self.d_concept_de, self.d_fea)
        if "gru" in self.text_mode:
            self.num_feat += 1
            self.num_layers = 2
            self.bidirectional = True
            print("GRU Embedding: %d" % self.d_concept_en)
            self.embedding = nn.Embedding(self.d_concept_en, self.d_fea)
            self.gru_encoder = nn.GRU(self.d_fea, self.d_fea, self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        if "cnn" in self.text_mode:
            self.frame_cnn = TextCNN(opts.visual_kernel_num, self.rnn_output_size, opts.visual_kernel_sizes, dropout=0.8)

        if self.num_feat > 1:
            self.feat_fuse_mlp = nn.Linear(self.d_fea * self.num_feat, self.d_fea)

    def visual_feat_encode(self, moment_feats):
        moment_feats = self.visual_feat_mlp(moment_feats)
        return moment_feats

    def query_encode(self, input_ids_bert, input_mask_bert, segment_ids_bert, tok_ids, bow_vec):
        query_feat = []
        if "bert" in self.text_mode:
            feat = self.bert_encode(input_ids_bert, input_mask_bert, segment_ids_bert).mean(dim=1)
            query_feat.append(feat)
        if "bow" in self.text_mode:
            feat = self.bow_encode(bow_vec)
            query_feat.append(feat)
        if "gru" in self.text_mode:
            feat = self.gru_encode(tok_ids)
            query_feat.append(feat)

        if self.num_feat > 1:
            query_feat = torch.cat(query_feat, dim=1)
            query_feat = self.feat_fuse_mlp(query_feat)
        else:
            query_feat = query_feat[0]
        return query_feat

    def bert_encode(self, input_ids, input_mask, segment_ids):
        outputs = self.bert_encoder(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        return outputs.last_hidden_state

    def bow_encode(self, bow_vec):
        outputs = self.bow_encoder(bow_vec)
        return outputs

    def gru_encode(self, text_ids):
        batch_size, sequence_length = text_ids.shape
        text_emb = self.embedding(text_ids)
        output, hn = self.gru_encoder(text_emb)
        if self.bidirectional:
            output = output.reshape(batch_size, sequence_length, 2, self.d_fea)
            output = output.mean(dim=2, keepdim=False)
        mean_output = output.mean(dim=1, keepdim=False)
        return mean_output


class DeepVR(RetrievalBaseModel):
    def __init__(self, d_fea, d_concept_en, d_concept_de, opts):
        super(DeepVR, self).__init__(opts, d_fea, d_concept_en, d_concept_de)
        self.training = False
        self.concept_decoder = nn.Linear(d_fea, d_concept_de)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids_bert, input_mask_bert, segment_ids_bert, tok_ids, query_bow, moment_feats):
        query_feat = self.query_encode(input_ids_bert, input_mask_bert, segment_ids_bert, tok_ids, query_bow)
        moment_feats = self.visual_feat_encode(moment_feats)
        concept_pred = self.decode(moment_feats)
        return moment_feats, query_feat, concept_pred

    def decode(self, moment_feats):
        concepts_pred = self.concept_decoder(moment_feats)
        concepts_pred = self.sigmoid(concepts_pred)
        return concepts_pred

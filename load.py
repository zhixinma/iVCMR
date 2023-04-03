import torch
import h5py
from util import padding
import numpy as np
from transformers import BertTokenizer
import networkx as nx
import random
import os
from util import mean

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_CAP_LEN = 32


# global setting
def load_query_input(query_id, input_reader, vocab):
    """
    :param query_id:
    :param input_reader: LMDB file
    :param vocab:
    :return:
    """
    query_data = input_reader[query_id]
    if query_id == "95770":
        _ = 1
    query_text = query_data["desc"]
    query_id = int(query_data["desc_id"]) if "desc_id" in query_data else int(query_id)
    start_id = query_data["start"]
    tar_node_ids = torch.tensor(query_data["target"])
    bow_vec = vocab.text_to_bow(query_text)
    tok_ids = vocab.text_to_tok_id(query_text)
    input_ids_bert, input_mask, segment_ids = text_to_id_bert(query_text)

    # only for real scenarios
    rank = query_data["rank"] if "rank" in query_data else 0
    distance = query_data["distance"] if "distance" in query_data else 0
    proposal = query_data["proposal"] if "proposal" in query_data else []
    distance_proposal = query_data["distance_proposal"] if "distance_proposal" in query_data else []
    return query_id, start_id, tar_node_ids, query_text, input_ids_bert, input_mask, segment_ids, bow_vec, tok_ids, rank, distance, proposal, distance_proposal


def load_query_input_vr(query_id, input_reader, moment_feat_reader, vocab):
    query_data = input_reader[query_id]
    query_text = query_data["desc"]
    bow_vec = vocab.text_to_bow(query_text)
    tok_ids = vocab.text_to_tok_id(query_text)
    input_ids, input_mask, segment_ids = text_to_id_bert(query_text)
    query_feat = moment_feat_reader[query_id]
    return input_ids, input_mask, segment_ids, query_feat, bow_vec, tok_ids


def text_to_id_bert(text):
    tokens = ["[CLS]"] + bert_tokenizer.tokenize(text) + ["[SEP]"]
    tok_len = min(len(tokens), MAX_CAP_LEN)
    tokens_pad = padding(tokens, MAX_CAP_LEN, "[PAD]")
    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens_pad)
    input_mask = [1] * tok_len + [0] * (MAX_CAP_LEN-tok_len)
    segment_ids = [0] * tok_len + [0] * (MAX_CAP_LEN-tok_len)

    input_ids = torch.tensor(input_ids)
    input_mask = torch.tensor(input_mask)
    segment_ids = torch.tensor(segment_ids)
    return input_ids, input_mask, segment_ids


def assign_nk(graph: nx.Graph, nk_reader, sample_action, max_action_space):
    node_size = graph.number_of_nodes()
    ns_nodes = {}
    for i_node in range(node_size):
        ns = nk_reader["%d" % i_node]
        if sample_action and len(ns) > max_action_space:
            random.seed(42)
            ns = random.sample(ns, max_action_space)
        ns_nodes[i_node] = {"ns": ns}
    nx.set_node_attributes(graph, ns_nodes)
    return graph


def load_graph(nk_reader, opts):
    if opts.use_balanced_graph:
        graph_path = "%s/%s_balanced_graph_ths_%.2f_thx_%.2f_wc_%.2f_ws_%.2f_tok_%d.gpickle" % (opts.graph_root, opts.dataset, opts.edge_threshold_single, opts.edge_threshold_cross, opts.weight_concept,
                                                                                                opts.weight_subtitle, opts.k_ei)
    else:
        graph_path = "%s/graph_ths_%.2f_thx_%.2f_wc_%.2f_ws_%.2f.gpickle" % (opts.graph_root, opts.edge_threshold_single, opts.edge_threshold_cross, opts.weight_concept, opts.weight_subtitle)
    print("Loading graph...: %s" % graph_path)
    assert os.path.isfile(graph_path), graph_path
    g = nx.read_gpickle(graph_path)
    print(nx.info(g))

    # Assign actions
    if opts.action_type == "nk":  # neighbors within k steps
        assert (nk_reader is not None) and (opts.max_action_space is not None)
        sample_action = (opts.mode == "train" and opts.sample_action)
        g = assign_nk(g, nk_reader, sample_action, opts.max_action_space)
        avg_action_size = mean([len(g.nodes[i]["ns"]) for i in range(len(g))])
        print("Action space: N%d Average Action Size: %.3f" % (opts.k_for_nk, avg_action_size))
    elif opts.action_type in ["n01", "n1"]:
        avg_action_size = mean([len(g[i]) for i in range(len(g))])
        print("Action space: N1 Average Action Size: %.3f" % avg_action_size)
    else:
        assert False, "Error: unsupported action type: %s." % opts.action_type
    return g


def load_concept_prob(prob_path):
    print("Loading probability: %s" % prob_path)
    with h5py.File(prob_path, "r") as f:
        prob = f["concept_prob"][()].astype(np.float32)
    return prob


def load_concepts(concept_path):
    with open(concept_path, "r") as f:
        concepts = f.read().split()
    concept_size = len(concepts)
    print("Load: concept size", concept_size)
    return concepts, concept_size


def load_video_id(video_id_path):
    with open(video_id_path, "r") as f:
        video_ids = f.read().split()
    video_size = len(video_ids)
    print("Load: video size", video_size)
    return video_ids, video_size

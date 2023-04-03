import argparse
from trainer import Trainer
from trainer_vr import TrainerVR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument("--prob_path", default="", type=str, help="")
    # parser.add_argument("--emb_path", default="", type=str, help="")
    # parser.add_argument("--concept_list_path", default="", type=str, help="")
    # parser.add_argument("--video_id_path", default="", type=str, help="")
    # parser.add_argument("--output_dir", default=None, type=str, help="")
    parser.add_argument("--feature_root", default="/path/to/feature", type=str, help="path to feature directory")
    parser.add_argument("--graph_root", default="/path/to/graph", type=str, help="path to graph directory")
    parser.add_argument("--data_root", default="/path/to/data", type=str, help="path to preprocessed data directory")
    parser.add_argument("--ckpt_root", default="/path/to/ckpt", type=str, help="the output directory where the model checkpoints will be written.")
    parser.add_argument("--dataset", choices=["tvr", "didemo"], default=None, type=str, help="The name of dataset")

    # Training Setting
    parser.add_argument("--split", choices=["val", "test_public", "test", "train"], default="train", type=str, help="The input query split")
    parser.add_argument('--toy', action='store_true', help="toy mode")
    parser.add_argument("--training_prefix", default="test", type=str, help="")
    parser.add_argument("--text_mode", type=str, default="bert,gru", help="")
    parser.add_argument('--freeze_bert', action='store_true', help="")
    parser.add_argument("--task", choices=["ivcml", "vr"], type=str, help="")
    parser.add_argument("--margin", default=0.1, type=float, help="")
    parser.add_argument("--early_stop_epoch_num", default=10, type=int, help="early stop if there is not improvement for the number of epochs.")
    parser.add_argument('--drop_graph', action='store_true', help="don't employ graph structure.")
    parser.add_argument('--no_feedback', action='store_true', help="don't consider feedback.")
    parser.add_argument('--random_policy', action='store_true', help="")
    parser.add_argument('--estimate_gold', action='store_true', help="estimate a ground truth if no gt appears in action space.")

    # Graph Setting
    parser.add_argument("--edge_threshold_single", type=float, help="The threshold of connecting to nodes.")
    parser.add_argument("--edge_threshold_cross", type=float, help="The threshold of connecting to nodes.")
    parser.add_argument("--weight_concept", default=0.8, type=float, help="")
    parser.add_argument("--weight_subtitle", default=0.2, type=float, help="")
    parser.add_argument("--alpha", default=0.89, type=float, help="decay rate")
    parser.add_argument("--init_lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--feature_type", default="vt", type=str, help="")

    # RL Setting
    parser.add_argument("--gamma", default=0.8, type=float, help="discount rate for value")
    parser.add_argument("--gae_lambda", default=1.0, type=float, help="discount rate for gae")
    parser.add_argument("--phi", default=0, type=float, help="penalty factor for time")
    parser.add_argument("--ent_coef", default=0.5, type=float, help="")
    parser.add_argument("--vf_coef", default=1, type=float, help="")
    parser.add_argument("--mode", choices=["train", "infer", "infer_hero", "infer_conquer"], default="train", type=str, help="")
    parser.add_argument("--select_by", choices=["path", "step"], default="path", type=str, help="")
    parser.add_argument("--loss_func", default="ce", type=str, help="")
    parser.add_argument("--loss_weight", default="1", type=str, help="")
    parser.add_argument("--resume", action='store_true', help="resume from checkpoint")
    parser.add_argument("--reward_signal", choices=["step", "path"], default="step", type=str, help="")
    parser.add_argument("--transition", choices=["policy", "value"], default="policy", type=str, help="use policy or value to step.")
    parser.add_argument("--infer_steps", default=4, type=int, help="")
    parser.add_argument("--max_action_space", default=300, type=int, help="The maximum action number. Only take effect when $(sample_action) is True")
    parser.add_argument("--sample_action", action='store_true', help="Sample $(max_action_space) actions if is True")
    parser.add_argument("--batch_size", default=64, type=int, help="The minimum N6 size of target moment.")
    parser.add_argument("--k_ei", default=10, type=int, help="k for top-k important edges")
    parser.add_argument('--use_balanced_graph', action='store_true', help="")
    parser.add_argument('--no_query_update', action='store_true', help="")
    parser.add_argument('--wo_replacement', action='store_true', help="")
    parser.add_argument('--vr_initialize', action='store_true', help="use vr task to initialize or not")
    parser.add_argument('--stem', action='store_true', help="stem token or not")

    # fixed
    parser.add_argument("--action_type", choices=["n01", "n1", "static_nk", "nk", "knn"], default="nk", type=str, help="")
    parser.add_argument("--k_for_nk", default=3, type=int, help="Depth limit for bfs.")
    args = parser.parse_args()

    # # fixed parameters
    # args.freeze_bert = True
    # args.vr_initialize = False  # pre

    if args.task == "ivcml":
        trainer = Trainer(args)
    elif args.task == "vr":
        trainer = TrainerVR(args)
    else:
        print("Unsupported Task: %s" % args.task)
        raise NotImplementedError

    trainer.run()

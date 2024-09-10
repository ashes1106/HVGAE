import argparse

datasets_args = {
    "dblp": {
        "type_num": [4057, 14328, 7723, 20],  # the number of every node type
        "nei_num": 1,  # the number of neighbors' types
        "n_labels": 4,
    },
    "aminer": {
        "type_num": [6564, 13329, 35890],
        "nei_num": 2,
        "n_labels": 4,
    },
    "freebase": {
        "type_num": [3492, 2502, 33401, 4459],
        "nei_num": 3,
        "n_labels": 3,
    },
    "acm": {
        "type_num": [4019, 7167, 60],
        "nei_num": 2,
        "n_labels": 3,
    },
}


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of hidden layers")

    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--feat_drop", type=float, default=.2,
                        help="input feature dropout")

    parser.add_argument("--norm", type=str, default=None)

    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")

    parser.add_argument("--replace_rate", type=float, default=0.0,
                        help="The replace rate. The ratio of nodes that is replaced by random nodes.")
    parser.add_argument("--leave_unchanged", type=float, default=0.3,
                        help="The ratio of nodes left unchanged (no mask), but is asked to reconstruct.")

    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")

    parser.add_argument("--alpha_l", type=float, default=2,
                        help="pow index for sce/esce loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99,
                        help="decay the lr by gamma for ExponentialLR scheduler")

    parser.add_argument('--ratio', type=int, default=[20, 40, 60])

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=1024)  # 64

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)  # 0.05
    parser.add_argument('--eva_wd', type=float, default=0, help="weight decay")

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=0)

    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument("--loss_fn", type=str, default="esce")
        
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='factor for contrastive loss')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='factor for kl loss')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='factor for recontruction loss')
    parser.add_argument("--lr", type=float, default=0.0004,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument("--feat_mask_rate", type=float,
                        default=0.3, help="feature mask rate")
    parser.add_argument("--dp_rate", type=float,
                        default=0.1, help="learning rate")
    parser.add_argument("--attn_drop", type=float,
                        default=0.3, help="attention dropout")
    parser.add_argument('--devi_num', type=float, default=3)

    parser.add_argument('--neg_num', type=int, default=10)

    # read config
    parser.add_argument("--use_cfg", action="store_true",
                        help="Set to True to read config file")

    parser.add_argument("--task", type=str, default="clustering",
                        choices=["classification", "clustering"])
    parser.add_argument('--debug', type=bool, default=True, help='-')

    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)
    return args

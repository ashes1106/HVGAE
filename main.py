import datetime
import sys
import warnings
import numpy as np
from collections import defaultdict
import torch

from hvgae.models.edcoder import PreModel
from hvgae.utils import (evaluate, evaluate_cluster, load_best_configs, load_data,
                          metapath2vec_train, preprocess_features,
                          set_random_seed)
from hvgae.utils.args import build_args
from tqdm import tqdm
from pprint import pprint
import logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


class Trainer():
    def __init__(self, args, model, epochs, idx_train, idx_val, idx_test, nb_classes, label, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.epochs = epochs
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.nb_classes = nb_classes
        self.label = label
        self.device = device

    def train(self, feats, mps,  nei_index, scheduler, optimizer, alpha, beta, gamma, dataset):
        best_model_state_dict = None
        cnt_wait = 0
        best = 1e9
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            optimizer.zero_grad()
            loss, loss_item = self.model(
                feats, mps, alpha, beta, gamma, self.epochs, nei_index=nei_index, epoch=epoch)
            if epoch % 5 == 0:
                logging.info(
                    f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}")
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                best_model_state_dict = self.model.state_dict()
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        message = self.eval(best_model_state_dict, feats,
                            mps, nei_index, dataset)
        return message

    def eval(self, best_model_state_dict, feats, mps, nei_index, dataset):
        self.model.load_state_dict(best_model_state_dict)
        self.model.eval()
        embeds = self.model.get_embeds(feats, mps, dataset, nei_index)
        mas_m_l, mas_s_l, mis_m_l, mis_s_l, auc_m_l, auc_s_l = [], [], [], [], [], []
        for i in range(len(self.idx_train)):
            mas_m, mas_s, mis_m, mis_s, auc_m, auc_s = evaluate(embeds, self.idx_train[i], self.idx_val[i],
                                                                self.idx_test[i], self.label, self.nb_classes,
                                                                self.device, self.args.eva_lr, self.args.eva_wd)

            mas_m_l.append(mas_m)
            mas_s_l.append(mas_s)
            mis_m_l.append(mis_m)
            mis_s_l.append(mis_s)
            auc_m_l.append(auc_m)
            auc_s_l.append(auc_s)

        idxmap = {0: "20", 1: "40", 2: "60"}
        message = defaultdict(float)
        for idx, v in enumerate(zip(mas_m_l, mas_s_l, mis_m_l, mis_s_l, auc_m_l, auc_s_l)):
            t = str(idxmap[idx])
            message[t+"_mas_mean"] = round(v[0], 4)
            message[t+"_mas_std"] = round(v[1], 4)
            message[t+"_mis_mean"] = round(v[2], 4)
            message[t+"_mis_std"] = round(v[3], 4)
            message[t+"_auc_mean"] = round(v[4], 4)
            message[t+"_auc_std"] = round(v[5], 4)

        return message


def main():
    args = build_args()
    device = torch.device("cuda:{}".format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    # random seed
    set_random_seed(args.seed)
    # load data
    (nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]

    num_mp = int(len(mps))
    print("Dataset: ", args.dataset)

    # main params
    lr = args.lr
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    epochs = args.epochs
    dp_rate = args.dp_rate
    feat_mask_rate = args.feat_mask_rate
    devi_num = args.devi_num
    attn_drop = args.attn_drop
    neg_num = args.neg_num

    focused_feature_dim = feats_dim_list[0]
    model = PreModel(args, num_mp, focused_feature_dim, devi_num,
                     neg_num, dp_rate, feat_mask_rate, attn_drop)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=args.l2_coef)
    # scheduler
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    model.to(device)
    feats = [feat.to(device) for feat in feats]
    mps = [mp.to(device) for mp in mps]
    label = label.to(device)
    idx_train = [i.to(device) for i in idx_train]
    idx_val = [i.to(device) for i in idx_val]
    idx_test = [i.to(device) for i in idx_test]

    trainer = Trainer(args, model, epochs, idx_train, idx_val,
                      idx_test, nb_classes, label, device)
    message = trainer.train(feats, mps, nei_index, scheduler,
                            optimizer,  alpha, beta, gamma, args.dataset)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    main()

from itertools import chain
from functools import partial
import torch
import torch.nn as nn
import dgl
from dgl import DropEdge

from hgcvae.models.loss_func import sce_loss, esce_loss
from hgcvae.utils import create_norm, grace_loss, info_nce_loss_torch
from hgcvae.models.han import HAN

import logging
logging.basicConfig(level=logging.INFO)

class PreModel(nn.Module):
    def __init__(
            self,
            args,
            num_metapath: int,
            focused_feature_dim: int,
            devi_num: int,
            neg_num: int,
            dp_rate: float,
            feat_mask_rate: float,
            attn_drop: float
    ):
        super(PreModel, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        # self.feat_mask_rate = args.feat_mask_rate
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn
        self.enc_dec_input_dim = self.focused_feature_dim

        self.attn_drop = attn_drop
        self.devi_num = devi_num
        self.neg_num = neg_num
        self.dp_rate = dp_rate

        self.feat_mask_rate = feat_mask_rate

        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # num head: encoder
        if self.encoder_type in ("gat", "dotgat", "han"):
            enc_num_hidden = self.hidden_dim // self.num_heads
            enc_nhead = self.num_heads
        else:
            enc_num_hidden = self.hidden_dim
            enc_nhead = 1

        # num head: decoder
        if self.decoder_type in ("gat", "dotgat", "han"):
            dec_num_hidden = self.hidden_dim // self.num_out_heads
            dec_nhead = self.num_out_heads
        else:
            dec_num_hidden = self.hidden_dim
            dec_nhead = 1
        dec_in_dim = self.hidden_dim

        # encoder
        self.encoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_dec_input_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            devi_num=self.devi_num,
            neg_num=self.neg_num,
            dp_rate=self.dp_rate,
            norm=self.norm,
        )

        # decoder
        self.decoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            nhead=enc_nhead,
            nhead_out=dec_nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            devi_num=self.devi_num,
            neg_num=self.neg_num,
            dp_rate=self.dp_rate,
            norm=self.norm,
            concat_out=True,
        )
        
        self.alpha_l = args.alpha_l
        self.attr_restoration_loss = self.setup_loss_fn(
            self.loss_fn, self.alpha_l)
        self.__cache_gs = None
        self.enc_mask_token = nn.Parameter(
            torch.zeros(1, self.enc_dec_input_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self._replace_rate = args.replace_rate
        self._leave_unchanged = args.leave_unchanged
        assert self._replace_rate + \
            self._leave_unchanged < 1, "Replace rate + leave_unchanged must be smaller than 1"
         
    @property
    def output_hidden_dim(self):
        return self.hidden_dim
 

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "esce":
            criterion = partial(esce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self._leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[: num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
            :num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_restoration(self, feat, gs, alpha, beta, gamma, epochs, epoch):        
        cur_feat_mask_rate = self.feat_mask_rate
        mask_step = round(self.feat_mask_rate / epochs, 4)
        cur_feat_mask_rate = max(self.feat_mask_rate + mask_step * epoch, 0)        

        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            feat, cur_feat_mask_rate)
        
        enc_out1, enc_out2, _, _, _, mean, log_std, negs = self.encoder(
            gs, use_x, epoch, return_hidden=False)

        neg_batch = torch.cat(negs)
        
        info_loss = info_nce_loss_torch(
            enc_out1, enc_out2, neg_batch, self.hidden_dim, temperature=0.1)        

        kl_loss = -0.5 * (1 + 2 * log_std - mean ** 2 -
                          torch.exp(log_std) ** 2).sum(1).mean()        
        enc_out_mapped = self.encoder_to_decoder(enc_out2)

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(enc_out_mapped)            
        else:
            feat_recon, att_mp, _, _, _, _ = self.decoder(
                gs, enc_out_mapped, epoch)
            

        if cur_feat_mask_rate == 0.0:
            rec_loss = self.attr_restoration_loss(feat_recon, feat)
        else:
            x_init = feat[mask_nodes]
            x_rec = feat_recon[mask_nodes]
            rec_loss = self.attr_restoration_loss(x_rec, x_init)
        
        loss = alpha * info_loss + beta * kl_loss + gamma * rec_loss
        # if epoch % 5 == 0:
        #     logging.info("Epoch {:05d} |  Loss {:.4f}".format(epoch, loss.item()))            

        return loss, feat_recon, att_mp, enc_out1, mask_nodes
 

    def get_embeds(self, feats, mps, dataset, *varg):        
        origin_feat = feats[0]
        gs = self.mps_to_gs(mps)
        rep, _, _, _, _, _, _, _ = self.encoder(gs, origin_feat, -1)        
        return rep.detach()

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            for mp in mps:
                indices = mp._indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                gs.append(cur_graph)
            return gs
        else:
            return self.__cache_gs
        
    def forward(self, feats, mps, alpha, beta, gamma, epochs,  **kwargs):                
        origin_feat = feats[0]        
        gs = self.mps_to_gs(mps)
        loss, feat_recon, att_mp, enc_out, mask_nodes = self.mask_attr_restoration(origin_feat, gs, alpha, beta, gamma, epochs, kwargs.get("epoch", None))                
        return loss, loss.item()


def setup_module(m_type, num_metapath, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual,
                 norm, nhead, nhead_out, attn_drop, devi_num, neg_num, dp_rate, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "han":
        mod = HAN(
            num_metapath=num_metapath,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            enc_dec=enc_dec,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            devi_num=devi_num,
            neg_num=neg_num,
            dp_rate=dp_rate,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod

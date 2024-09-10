from typing import List
import torch
import torch.nn as nn

import dgl
from hgcvae.models.gat import GATConv
from hgcvae.utils import create_activation
import torch.nn.functional as F
import random

from hgcvae.utils.args import build_args
args = build_args()
device = torch.device("cuda:" + str(args.gpu))

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z).sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp


class HANLayer(nn.Module):

    def __init__(self, num_metapath, in_dim, out_dim, nhead,
                 feat_drop, attn_drop, negative_slope, residual, activation, norm, concat_out):
        super(HANLayer, self).__init__()
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(
                in_dim, out_dim, nhead,
                feat_drop, attn_drop, negative_slope, residual, activation, norm=norm, concat_out=concat_out))
        self.semantic_attention = SemanticAttention(in_size=out_dim * nhead)

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, new_g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))  # flatten because of att heads
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        out, att_mp = self.semantic_attention(semantic_embeddings)  # (N, D * K)

        return out, att_mp


class HAN(nn.Module):
    def __init__(self,
                 num_metapath,
                 in_dim,
                 num_hidden,
                 out_dim,
                 enc_dec,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 devi_num, 
                 neg_num,
                 dp_rate, 
                 concat_out=False,
                 encoding=False                 
                 ):
        super(HAN, self).__init__()        
        self.num_heads = nhead
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.activation = create_activation(activation)
        self.concat_out = concat_out
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.enc_dec=enc_dec
        self.devi_num = devi_num
        self.neg_num = neg_num
        self.dp_rate = dp_rate
        

        last_activation = create_activation(activation) if encoding else create_activation(None)
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.han_layers.append(HANLayer(num_metapath, in_dim, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual, last_activation,
                                            norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.han_layers.append(HANLayer(num_metapath, 
                                            in_dim, num_hidden, nhead,
                                            feat_drop, attn_drop, negative_slope, residual, self.activation, norm=norm,
                                            concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # in_dim = num_hidden * num_heads
                self.han_layers.append(HANLayer(num_metapath, 
                                                num_hidden * nhead, num_hidden, nhead,
                                                feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                norm=norm, concat_out=concat_out))
            # output projection
            self.han_layers.append(HANLayer(num_metapath, 
                                            num_hidden * nhead, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm, concat_out=concat_out))
        


        self.mean_mlp = HANLayer(num_metapath, 
                                            num_hidden * nhead, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm, concat_out=concat_out)
        
        self.std_mlp = HANLayer(num_metapath, 
                                            num_hidden * nhead, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm, concat_out=concat_out)
 
        self.decoder = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())        




    def forward(self, gs: List[dgl.DGLGraph], h, epoch, return_hidden=False):        
        h1,h2=h,h
        if self.enc_dec == "encoding":
            for gnn in self.han_layers:
                h1, att_mp = gnn(gs, h1)            

            for gnn in self.han_layers:
                h2, att_mp = gnn(gs, h2)                        

            mean, _ = self.mean_mlp(gs, h1)
            log_std, _ = self.std_mlp(gs, h1)
            
            mean = F.normalize(mean, dim=-1)
            log_std = F.normalize(log_std, dim=-1)
            
            gaussian_noise1 = torch.randn(h.size(0), self.out_dim * self.num_heads).to(device)
            gaussian_noise2 = torch.randn(h.size(0), self.out_dim * self.num_heads).to(device)
            z1 = mean + gaussian_noise1 * torch.exp(log_std)
            z2 = mean + gaussian_noise2 * torch.exp(log_std)
            
            negs = []
            neg_step = self.neg_num * 2 / 25
            vae_neg_nums = int(neg_step* epoch)
            encode_neg_nums = int(self.neg_num * 2) - vae_neg_nums            
            for _ in range(encode_neg_nums):                
                negs.append(F.dropout(h1, self.dp_rate))


            devi_num = self.devi_num 
            mean_neg = mean * devi_num
            log_std_neg = log_std
            for _ in range(vae_neg_nums):
                g_noise = torch.randn(h.size(0), self.out_dim * self.num_heads).to(device)
                neg_z =  mean_neg + g_noise * torch.exp(log_std_neg)
                negs.append(neg_z)

            random.shuffle(negs)
            return h1, h2, att_mp, z1, z2, mean, log_std, negs
        else:            
            for gnn in self.han_layers:
                h, att_mp = gnn(gs, h)                  
            return h, None, None, None, None, None
 

        

def add_noise(tensor, noise_ratio):    
    tensor_mean = tensor.mean(dim=0)
    tensor_std = tensor.std(dim=0)
    noise = torch.randn_like(tensor)
    noise_std = noise_ratio * tensor_std
    noisy_tensor = tensor + noise * noise_std
    return noisy_tensor


    
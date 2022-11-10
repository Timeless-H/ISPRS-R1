import sys 
sys.path.append('./')

import torch
import torch.nn as nn
import math
from preformer.helper_files.tools import Config as cfg

# class ln():
#     global layer_num
#     layer_num = 0

class SoftmaxAttention(nn.Module):
    def __init__(self,):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(cfg.dropout)
        self.head_dim = cfg.head_dim

    def forward(self, Q, K, V, mask):
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X


class NystromAttention(nn.Module):
    def __init__(self, layer_num, use_conv=False):
        super().__init__()
        self.layer_num = layer_num
        self.use_conv = use_conv
        self.head_dim = cfg.head_dim[layer_num]
        self.num_head = cfg.num_heads[layer_num]

        self.num_landmarks = cfg.num_landmarks[layer_num]
        self.seq_len = cfg.seq_len[layer_num]

        # self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (cfg.conv_kernel_size, 1), padding = (cfg.conv_kernel_size // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):
        # print("Qshape: {}  ||  mask: {}".format(Q.shape, mask[:, None, :, None].shape))
        Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])

        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim = -2)) * torch.max(torch.sum(torch.abs(K), dim = -1))) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'


class Attention(nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()

        # note dim = num_heads * head_dim
        self.layer_num = layer_num
        self.dim = dim
        self.head_dim = cfg.head_dim[layer_num]
        self.num_head = cfg.num_heads[layer_num]

        self.attn_type = cfg.attn_type

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention()

        elif self.attn_type == "nystrom":
            self.attn = NystromAttention(layer_num)


        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask):

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        with torch.cuda.amp.autocast(enabled = False):
            attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        return out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


class Transformer_Layer(nn.Module):
    def __init__(self, dim, layer_num, use_mask=True):
        super().__init__()
        self.layer_num = layer_num
        self.use_mask = use_mask
        self.dim = dim
        self.hidden_dim = dim * 4

        self.mha = Attention(dim, layer_num)

        self.dropout1 = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(self.dim)

        self.ff = torch.nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )

        self.dropout2 = torch.nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, X):
        X = X.permute(0, 2, 1)
        if self.use_mask:
            B, N, _ = X.shape
            mask = torch.ones(B, N).bool().cuda()
        mha_out = self.norm1(X + self.dropout1(self.mha(X, mask)))
        mha_out = self.norm2(mha_out + self.dropout2(self.ff(mha_out)))
        return mha_out.permute(0,2,1)


def main():
    cloud = torch.randn(16, 4096, 64)
    

    nymo = Transformer_Layer(dim=64)
    pred = nymo(cloud)

    print("pred_shape: {}".format(pred.shape))


if __name__=="__main__":
    main()
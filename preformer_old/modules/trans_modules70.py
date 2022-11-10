import torch, math
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F
from preformer.helper_files.tools import Config

pnum_dict1 = {Config.num_points:0, int(Config.num_points/4):1, int(Config.num_points/16):2, int(Config.num_points/64):3}
pnum_dict = {4096:0, 1024:1, 256:2, 64:3}

class sth2():
    global n_idx
    global sub_idx
    n_idx = []
    sub_idx = []


def nbrhood_gathering(var, idx):
    """
    :param var: [B, C, N]; features of the point cloud
    :param idx: neighbor indexes
    --------
    :return: [B, C, N, K]  # not sure of shape yet
    """

    # finding neighboring points
    _, _, C = var.shape
    B, N, K = idx.shape
    extended_idx = idx.unsqueeze(1).expand(B, C, N, K) # create axis, repeat [N,K] content on new axis
    extended_xyz = var.transpose(-2,-1).unsqueeze(-1).expand(B, C, N, K)  # .expand: repeatition along newly created axis
    neighbors = torch.gather(extended_xyz, 2, extended_idx)  # shape (B, C, N, K)
    return neighbors.permute(0, 2, 3, 1)  # shape (B, N, K, C)


class PointPositionEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim, device):
        super(PointPositionEmbedding, self).__init__()
        self.device = device
        self.mlp1 = nn.Sequential(
            nn.Linear(10, hidden_dim),  # no 4
            nn.ReLU(),
            nn.Linear(hidden_dim, dim))  # no 4; dim


    def forward(self, xyz, idx, num_neighbors=16):
        """
            xyz: [B, N, 3];  xyz coordinates of the point cloud
            feats: [B, C, N, 1]; features of the point cloud
            knn_output: a tuple of neighbor indexes and distances
            -------
            Returns [B, 2*C, N, K]  # not sure of shape yet
        """

        # finding neighboring points
        # dist, idx = distNindex(xyz, num_neighbors)  # dist: [B N K]; idx: [B N K]
        # idx = sth2.n_idx
        dist = sth2.dist[pnum_dict[idx.shape[1]]]
        B, N, K = idx.shape
        #todo: the 3 down here shd be hidden_dim for non-xyz operations

        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K) # create axis, repeat [N,K] content on new axis
        extended_xyz = xyz.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)  # .expand: repeatition along newly created axis
        # extended_idx1 = idx.unsqueeze(1).expand(B, 3, N, K) # create axis, repeat [N,K] content on new axis
        neighbors = torch.gather(extended_xyz, 2, extended_idx)  # shape (B, 3, N, K)
        # neighbors1 = torch.gather(extended_xyz, 2, extended_idx1)  # shape (B, 3, N, K)
        # print(torch.equal(neighbors, neighbors1), torch.equal(dist, dist_1))

        # relative point position encoding
        concat = torch.cat((extended_xyz, neighbors, extended_xyz - neighbors,
                            dist.unsqueeze(-3)), dim=-3).type(torch.FloatTensor).to(self.device)
        # feats = self.mlp1(feats)
        # return torch.cat((self.mlp2(concat), feats.expand(B, -1, N, K)), dim=-3)
        return self.mlp1(concat.permute(0, 2, 3, 1))


class fast_attn(nn.Module):
    def __init__(self, dim, layer_num, hidden_dim=64, num_nbrs=16, share_kv=False):
        super().__init__()
        self.layer_num = layer_num
        self.num_nbrs = num_nbrs
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.share_kv = share_kv
        # todo: MLP: linear, gelu, drop, linear, drop
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),  # todo: test GELU scenario
            nn.Linear(dim * 4, dim))
        self.pos_embedding = PointPositionEmbedding(dim, hidden_dim, device='cuda:0')

    def forward(self, pos, x):
        # todo: norm x -> attn -> drop -> shortcut; 
        # then norm abv output -> mlp -> drop -> shortcut

        x = x.permute(0, 2, 1)
        B, N, _ = x.shape

        Q, K, V = self.to_qkv(x).chunk(3, dim=-1)  # [B N C] per

        idx = sth2.n_idx[pnum_dict[Q.shape[1]]]  # .to(x.device)
        # print('transformer new shape: {}'.format(Q.shape))
        # starting = time.time()
        # _, idx = distNindex(pos, self.num_nbrs)  # dist: [B N K]; idx: [B N K]
        # print('existing and new n_idx outputs: {} | duration: {} seconds'.format(torch.equal(idx, idx_1), time.time() - starting))

        Q = nbrhood_gathering(Q, idx)  # shape (B, N, K, C=dim)
        K = nbrhood_gathering(K, idx)  # todo: replace
        V = nbrhood_gathering(V, idx)

        pos_emb = self.pos_embedding(pos, idx, self.num_nbrs)  # shape (B, N, K, C=dim)

        kv_context = F.normalize(K+pos_emb, p=2, dim=-1).permute(0, 2, 3, 1) @ F.relu(V+pos_emb, inplace=True).permute(0, 2, 1, 3)  # [B K C C]
        Q_norm = F.normalize(Q, p=2, dim=-1).permute(0, 2, 1, 3)
        agg = self.mlp(torch.sum((1 / N) * Q_norm @ kv_context, dim=1)) + x  # [B N C]
        return agg.permute(0, 2, 1)


class IT_Fast_Attn(nn.Module):
    def __init__(self, dim, layer_num, hidden_dim=64, niter_heads=2):
        super(IT_Fast_Attn, self).__init__()
        self.layer_num = layer_num
        self.niter_heads = niter_heads
        self.shared_qkv = nn.Linear(dim, dim, bias=False)
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(dim, dim * 4),
        #     nn.ReLU(inplace=True),  # todo: test GELU scenario
        #     nn.Linear(dim * 4, dim)
        # )
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=True),
            nn.ReLU(inplace=True),  # todo: test GELU scenario
        )
        # self.proj = nn.Linear(dim, dim * 4, bias=False)
        self.drop1 = nn.Dropout(0.2)
        self.lnorm1 = nn.LayerNorm(4 * dim)
        self.lnorm2 = nn.LayerNorm(dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(4 * dim, dim, bias=False),
            nn.Dropout(0.2)
        )
        # self.pos_embedding = PointPositionEmbedding(dim, hidden_dim, device='cuda:0')


    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, N, C = x.shape  # assuming C = 32

        x = self.shared_qkv(x)  # i.e., q k and v all share the same weight
        # print(pnum_dict1)
        # idx = sth2.n_idx[pnum_dict1[x.shape[1]]]
        # print(idx.shape)

        # xk = nbrhood_gathering(x, idx)  # shape (B, N, K, C=dim)
        v = x.clone().detach()

        # pos_emb = self.pos_embedding(pos, idx)  # shape (B, N, K, C=dim)

        # first_context = F.normalize(xk + pos_emb, p=2, dim=-1).permute(0, 2, 3, 1) @ F.relu(vk + pos_emb, inplace=True).permute(0, 2, 1, 3)  # [B K C C]
        # out = self.mlp1(torch.sum((1 / N) * F.normalize(xk, p=2, dim=-1).permute(0, 2, 1, 3) @ first_context, dim=1))  # B N C

        for i in range(self.niter_heads):
            # if not i == 0:
            #     vk = nbrhood_gathering(v, idx)  # shape (B, N, K, C=dim)
            # first_context = xk.permute(0, 2, 3, 1) @ vk.permute(0, 2, 1, 3)  # [B K C C]  k.v
            first_context = x.permute(0, 2, 1) @ v  # [B C C]  k.v
            # print(first_context.shape, x.shape)
            # out = einsum('b n k c, b i k c -> b n c', xk, first_context.permute(0, 2, 1, 3))
            #mlp of bckc of first_context; softmax k of bckc; then einsum of both as out. NB: vk[:,:,0,:] shd == v
            out = F.softmax(x, dim=-1) @ first_context  # B N C q.kv
            # out = x @ first_context  # B N C q.kv
            out_cat1 = torch.cat([out, v], dim=2)  # B N C'  i.e., C' = 64  k[:, :, 0, :]
            out_mlp = self.mlp2(out_cat1)  # B N C'
            out_cat2 = torch.cat([out_cat1, out_mlp], dim=2)  # B N C''  i.e., c' = 128
            v = self.residual_mlp(self.lnorm1(self.drop1(out_cat2))) + x  # B N C
            v = self.lnorm2(x + v)

        return v.permute(0, 2, 1)

class ClassicMHA(nn.Module):
    def __init__(self, dim, layer_num, use_mask=False):
        super().__init__()
        self.layer_num = layer_num
        self.use_mask = use_mask
        self.dim = dim
        self.hidden_dim = dim * 4
        self.head_dim = Config.head_dim[layer_num]
        self.num_head = Config.num_heads[layer_num]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.drop_attn = nn.Dropout(0.2)

        self.ff_attn = nn.Linear(self.num_head * self.head_dim, self.dim)

        self.drop1 = nn.Dropout(0.2)
        self.lnorm1 = nn.LayerNorm(self.dim)

        self.ff_mha = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim),
        )

        self.drop2 = nn.Dropout(0.2)
        self.lnorm2 = nn.LayerNorm(self.dim)

    def forward(self, x, mask=None):

        x = x.permute(0, 2, 1)
        if self.use_mask:
            B, N, _ = x.shape
            mask = torch.ones(B, N).bool().cuda()
        
        # softmax scaled dot product with (multiple) heads
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # # kvq version of mha
        dot = torch.matmul(torch.transpose(K, -2, -1), V)
    
        # # qkv version of mha
        # dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        # dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = F.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        x = torch.matmul(Q, attn)
        # x = torch.matmul(attn, V)

        x = self.combine_heads(x)

        x = self.ff_attn(x)  # mha output

        # applying identity, normalization and dropout
        x = self.lnorm1(x + self.drop1(x))
        x = self.lnorm2(x + self.drop2(self.ff_mha(x)))
        
        return x.permute(0,2,1)

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


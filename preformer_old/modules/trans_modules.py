import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F


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

        self.mlp1a = nn.Linear(dim, dim*2)
        self.mlp1b = nn.Linear(dim*2, dim)
        self.bn1 = nn.BatchNorm2d(dim*2)

        self.mlp2 = nn.Linear(dim*2, dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.mlp3 = nn.Linear((dim*2)+dim, dim)
        self.bn3 = nn.BatchNorm1d(dim)
        # self.relu = nn.ReLU(inplace=True)

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(dim, dim*2),
        #     nn.BatchNorm2d(dim*2),  #goes with bcn n not bnc so figure out the arrangement
        #     nn.ReLU(inplace=True),  # todo: test GELU scenario
        #     nn.Linear(dim*2, dim)
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(2 * dim, dim, bias=True),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(inplace=True),  # todo: test GELU scenario
        # )
        # self.mlp3 = nn.Sequential(
        #     nn.Linear((dim*2)+dim, dim, bias=True),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(inplace=True),
        #)
        self.drop1 = nn.Dropout(0.2)
        self.lnorm1 = nn.LayerNorm(dim)
        self.lnorm2 = nn.LayerNorm(dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim, dim*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(dim*2, dim, bias=False)
        )
        # self.pos_embedding = PointPositionEmbedding(dim, hidden_dim, device='cuda:0')


    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, N, C = x.shape  # assuming C = 32
        x = self.shared_qkv(x)  # i.e., q k and v all share the same weight

        # idx = sth2.n_idx
        # xk = nbrhood_gathering(x, sth2.n_idx)  # shape (B, N, K, C=dim)  # N in n_idx must match N in x, else consider that of sub_idx
        # print('x: {}  || xk: {}'.format(x.grad_fn, xk.grad_fn))
        v = x.clone().detach()

        # pos_emb = self.pos_embedding(pos, idx)  # shape (B, N, K, C=dim)

        # first_context = F.normalize(xk + pos_emb, p=2, dim=-1).permute(0, 2, 3, 1) @ F.relu(vk + pos_emb, inplace=True).permute(0, 2, 1, 3)  # [B K C C]
        # out = self.mlp1(torch.sum((1 / N) * F.normalize(xk, p=2, dim=-1).permute(0, 2, 1, 3) @ first_context, dim=1))  # B N C

        for i in range(self.niter_heads):
            # if not i == 0:
            #     vk = nbrhood_gathering(v, sth2.n_idx)  # shape (B, N, K, C=dim)
            first_context = F.softmax(x, dim=-1).permute(0, 2, 1) @ v  # [B C C]  k.v
            # first_context = xk.permute(0, 2, 3, 1) @ vk.permute(0, 2, 1, 3)  # [B K C C]  k.v
            # first_context = self.mlp1b(F.relu(self.bn1(self.mlp1a(first_context.permute(0, 2, 1, 3)).permute(0, 3, 1, 2))).permute(0,2,3,1))  # mlp of bckc rather than bkcc
            # out = einsum('b i k c, b n k c -> b n c', first_context, xk)
            # mlp of bckc of first_context; softmax k of bckc; then einsum of both as out. NB: vk[:,:,0,:] shd == v
            # out = F.softmax(x, dim=-1) @ first_context  # B N C q.kv
            out = x @ first_context  # B N C q.kv
            # ssss = (x==vk[:, :, 0, :]).shape
            out_cat1 = torch.cat([out, v], dim=2)
            # out_cat1 = torch.cat([out, vk[:, :, 0, :]], dim=2)  # B N C'  i.e., C' = 128
            out_mlp = F.relu(self.bn2(self.mlp2(out_cat1).permute(0,2,1))).permute(0,2,1)  # B N C'
            out_cat2 = torch.cat([out_cat1, out_mlp], dim=2)  # B N C''  i.e., c' = 128+64
            v = self.lnorm1(self.mlp3(self.drop1(out_cat2)) + x)  # B N C
        
        v = self.lnorm2(self.residual_mlp(v) + v)  # in loop or out?

        return v.permute(0, 2, 1)


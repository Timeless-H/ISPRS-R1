import torch

class sth3():
    global sub_idx
    global neigh_idx
    sub_idx = []
    neigh_idx = []

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    ---------------------------------------------------
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def indexing_neighbor(tensor, index):
    """
    tensor: "(bs, vertice_num, dim)"
    index: "(bs, vertice_num, neighbor_num)"
    --------------------------------------------------------------
    Return: (bs, vertice_num, neighbor_num, dim) : grouped_xyz, but not ball queried
    """
    # todo: ball queried version
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index.long()]
    return tensor_indexed

def poolNsample(feat_map, layer_num: int):
    """
    feat_map: "(bs, vertex_num, channel_num)"
    """
    # if layer_num == 3:
    #     B, _, K = sth3.sub_idx.shape

    #     feature = feat_map.permute(0, 2, 1).contiguous()  # B N C
    #     neighbor_features = indexing_neighbor(feature, sth3.sub_idx)
    #     print("layer_3: bf: {}  ||  aft: {}".format(feature.shape, neighbor_features.shape))
    # else:
    #     B, _, K = sth3.neigh_idx.shape

    #     feature = feat_map.permute(0, 2, 1).contiguous()  # B N C
    #     neighbor_features = indexing_neighbor(feature, sth3.neigh_idx)
    #     pool_features = torch.max(neighbor_features, dim=2)[0]  # B N C
    #     print("layer_{}: bf: {}  ||  aft: {}".format(layer_num, feature.shape, neighbor_features.shape))

    '''reduce N from N to N' i.e., 4096 -> 1024 -> ... using sub_ind'''
    sampled_features = index_points(feat_map.permute(0, 2, 1).contiguous(), sth3.sub_idx[:, :, 0].long())  # B N' C
    # print("decimated: {}".format(sampled_features.shape))
    return sampled_features

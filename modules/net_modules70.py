import sys 
sys.path.append('./')

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional
from modules.trans_modules70 import fast_attn, IT_Fast_Attn, sth2, ClassicMHA
from helper_files.gen_utils import poolNsample, indexing_neighbor, sth3
from memcnn import InvertibleModuleWrapper  #todo: compare and correct
from memcnn.models.additive import AdditiveCoupling
from helper_files.tools import Config as cfg

CreateModuleFnType = Callable[[int, Optional[dict]], nn.Module]

NUM_LAYERS = 4
CHANNELS = [128, 128, 256, 384, 512]  # CHANNELS = [32, 32, 64, 128, 256]
ENCODER_DEPTH = [1, 1, 1, 1]
DECODER_DEPTH = [1, 1, 1, 1]
POS_DICT = {}
X_DICT = {}
POOL_CNT = 0

class sth():
    global iter_set
    iter_set = []

def getChannelsAtIndex(index: int):
    if index < 0: index = 0
    if index >= len(CHANNELS): index = len(CHANNELS) - 1
    return CHANNELS[index]

def getPosArgAtIndex(index: int):
    if index <= 0: key = 'pos'
    if index > 0: key = 'pos{}'.format(index)
    # if index >= len(POS_DICT):
    #     index = len(POS_DICT) - 1
    #     key = 'pos{}'.format(index)
    return POS_DICT[key]

def create_classicMHA_module(d_in, **kwargs):
    """pure MHA version"""
    layer_num = kwargs.get('layer_num')
    F_d_in = d_in // 2
    return AdditiveCouplingHalf(F=ClassicMHA(dim=F_d_in, layer_num=layer_num), channel_split_pos=F_d_in)

def create_fanet_module(d_in, **kwargs):  # F function concat split of input
    h_dim = kwargs.pop('h_dim', 64)
    layer_num = kwargs.get('layer_num')
    F_d_in = d_in // 2
    F_d_out = d_in - F_d_in

    return AdditiveCouplingHalf(F=IT_Fast_Attn(dim=F_d_in, layer_num=layer_num, hidden_dim=h_dim, niter_heads=2), channel_split_pos=F_d_out)

# def create_nystrom_module(d_in, **kwargs):
#     layer_num = kwargs.get('layer_num')
#     F_d_in = d_in // 2
#     return AdditiveCouplingHalf(F=Transformer_Layer(dim=F_d_in, layer_num=layer_num), channel_split_pos=F_d_in)

# def create_fanet_module(d_in, **kwargs):  # Double function rev. Fm, Gm
#     """Single Head Capacity"""
#     h_dim = kwargs.pop('h_dim', 64)
#     layer_num = kwargs.get('layer_num')
#     F_d_in = d_in // 2

#     return AdditiveCoupling(Fm=IT_Fast_Attn(dim=F_d_in, layer_num=layer_num, hidden_dim=h_dim, niter_heads=2), split_dim=1)

class AdditiveCouplingHalf(nn.Module):
    """Additive coupling layer, a basic invertible layer.
    By splitting the input activation :math:`x` and output activation :math:`y`
    into two groups of channels (i.e. :math:`(x_1, x_2) \cong x` and
    :math:`(y_1, y_2) \cong y`), `additive coupling layers` define an invertible
    mapping :math:`x \mapsto y` via
    .. math::
       y_1 &= x_2
       y_2 &= x_1 + F(x_2),
    where the `coupling function` :math:`F` is an (almost) arbitrary mapping.
    :math:`F` just has to map from the space of :math:`x_2` to the space of
    :math:`x_1`. In practice, this can for instance be a sequence of
    convolutional layers with batch normalization.
    The inverse of the above mapping is computed algebraically via
    .. math::
       x_1 &= y_2 - F(y_1)
       x_2 &= y_1.
    *Warning*: Note that this is different from the definition of additive
    coupling layers in ``MemCNN``. Those are equivalent to two consecutive
    instances of the above-defined additive coupling layers. Hence, the
    variant implemented here is twice as memory-efficient as the variant from
    ``MemCNN``.
    :param F:
        The coupling function of the additive coupling layer, typically a
        sequence of neural network layers.
    :param channel_split_pos:
        The index of the channel at which the input and output activations are
        split.
    """

    def __init__(self, F: nn.Module, channel_split_pos: int):
        super(AdditiveCouplingHalf, self).__init__()
        self.F = F
        self.channel_split_pos = channel_split_pos
        # self.pos_ref = []

    def forward(self, x):
        # pos = x[0][:, :3, :].permute(0, 2, 1)
        # x = x[0][:, 3:, :]
        x = x.permute(0, 2, 1)
        # print('pos: {} | x: {}'.format(pos.shape, x.shape))
        # x = channel_shuffle(x.unsqueeze(-1), groups=4).squeeze(-1)  # TODO: switch channel_shuffle on or off
        x1, x2 = x[:, :self.channel_split_pos], x[:, self.channel_split_pos:]
        x1, x2 = x1.contiguous(), x2.contiguous()
        y1 = x2  # todo: can expand to have revtorch version
        y2 = x1 + self.F.forward(x2)
        # print('pos: {} | x2: {}'.format(pos.shape, x2.shape))
        out = torch.cat([y1, y2], dim=1)
        # print('out: {} '.format(out.shape))
        # self.pos_ref = pos
        return out

    def inverse(self, y):  # pos,
        # y1, y2 = torch.chunk(y, 2, dim=1)
        inverse_channel_split_pos = y.shape[1] - self.channel_split_pos
        y1, y2 = (y[:, :inverse_channel_split_pos], y[:, inverse_channel_split_pos:])
        y1, y2 = y1.contiguous(), y2.contiguous()
        x2 = y1
        x1 = y2 - self.F.forward(y1)
        # print('backpass pos: {} | x1: {} | x2: {}'.format(self.pos_ref.shape, x1.shape, x2.shape))
        x = torch.cat([x1, x2], dim=1).permute(0, 2, 1)
        # print('x: {}'.format(x.shape))
        return x


class EncoderModule(nn.Module):
    def __init__(self, d_in: int, d_out: int, layer_num: int, depth: int=1, downsample: bool=True,
                 disable_custom_gradient: bool=False,  # todo: custom_gradient switch
                 create_module_fn: CreateModuleFnType = create_fanet_module):  # todo: switch btwn the different 'create *** modules
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        self.depth = depth
        self.poolCount = 0
        self.block_modules = nn.ModuleList()
        self.coordinate_kwargs = {
            'num_nbrs': None,
            'pos_mlp': 64,
            'attn_mlp': 4,
            'layer_num': layer_num
        }
        if downsample:
            self.conv = nn.Sequential(
                nn.Conv1d(d_in, d_out, 1, bias=True),
                nn.BatchNorm1d(d_out),
                nn.ReLU())
            d_in = d_out

        # create blocks in a layer
        # for i in range(depth):
        self.block_modules.append(
            InvertibleModuleWrapper(
                create_module_fn(
                    d_in, **self.coordinate_kwargs), 
                    disable=disable_custom_gradient
                )
            )
        # self.encoder_blocks = nn.ModuleList(self.block_modules)

    def forward(self, x):
        if self.downsample:
            global POOL_CNT
            if POOL_CNT == 0: X_DICT['x'] = x.clone().cpu()
            if POOL_CNT > 0: X_DICT['x{}'.format(POOL_CNT)] = x.clone().cpu()
            # if POOL_CNT > 2:
            sth3.sub_idx = sth.iter_set['sub_idx'][POOL_CNT]
            # sth3.neigh_idx = sth.iter_set['neigh_idx'][POOL_CNT]
            x_pool = poolNsample(x, layer_num=POOL_CNT)
            # x_pool_comp = x_pool.permute(0, 2, 1)
            POOL_CNT += 1
            # pos_pool, x_pool = pool_layer(pos, x)  # out
            # POS_DICT['pos{}'.format(POOL_CNT)] = pos_pool.clone().cpu()  # out
            # pos = pos_pool
            x = self.conv(x_pool.permute(0, 2, 1)).permute(0, 2, 1) #increase number of channels
            del x_pool
        # self.poolCount = POOL_CNT
        # self.coordinate_kwargs['layer_num'] = POOL_CNT
        # for i in range(self.depth):

            # print('Depth {}/{}'.format(i+1, self.depth))
            # data = torch.cat([pos.permute(0, 2, 1), x], dim=1)
            # print('encoder_main layer: {}'.format(POOL_CNT))
            # sth2.sub_idx = sth.iter_set['sub_idx'][POOL_CNT]
        sth2.n_idx = sth.iter_set['neigh_idx']  #[POOL_CNT]
            # sth2.dist = sth.iter_set['neigh_dist'] #[POOL_CNT]
        # print(self.downsample)
        x = self.block_modules[0](x)  # data --> x
        return x


class PyramidLevel1Block(nn.Module):
    def __init__(self, d_in: int, d_out: int, layer_num: int, save_var: str,
                disable_custom_gradient: bool=False,
                create_module_fn: CreateModuleFnType = create_fanet_module):
        super(PyramidLevel1Block, self).__init__()

        self.save_var = save_var
        self.coordinate_kwargs = {
            'num_nbrs': None,
            'pos_mlp': 64,
            'attn_mlp': 4,
            'layer_num': layer_num
        }

        self.conv1dim = nn.Sequential(nn.ConvTranspose1d(d_in, d_out, 1, bias=True),
                                      nn.BatchNorm1d(d_out),
                                      nn.ReLU())
        d_in = d_out

        self.PrevDec = InvertibleModuleWrapper(
            create_module_fn(d_in, **self.coordinate_kwargs), 
                disable=disable_custom_gradient
            )

    def forward(self, xx):
        pyr_map = {'210':['x1','x2'], '101':['x', '210'],}
        global POOL_CNT
        POOL_CNT -= 1
        if self.save_var == '210': 
            pre_x = X_DICT[pyr_map['210'][0]].to(xx.device)
            cur_x = X_DICT[pyr_map['210'][1]].to(xx.device)
        elif self.save_var == '101':
            pre_x = X_DICT[pyr_map['101'][0]].to(xx.device)
            cur_x = X_DICT[pyr_map['101'][1]].to(xx.device)

        # pooled_pos = sth.iter_set['xyz'][POOL_CNT]
        sth3.up_idx = sth.iter_set['interp_idx'][POOL_CNT]
        feat_map = indexing_neighbor(cur_x.permute(0, 2, 1), sth3.up_idx).squeeze(2)
        
        cur_x = torch.cat([pre_x, feat_map.permute(0, 2, 1)], dim=1)
        xx = self.conv1dim(cur_x).permute(0, 2, 1)
        # data = torch.cat([pooled_pos.permute(0, 2, 1), xx], dim=1)
        sth2.n_idx = sth.iter_set['neigh_idx']  #[POOL_CNT]

        X_DICT[self.save_var] = self.PrevDec(xx)  # data --> xx


class PyramidDecoderModule(nn.Module):
    def __init__(self, d_in: int, d_out: int, layer_num: int, upsample: bool=True, 
                disable_custom_gradient: bool=False,
                create_module_fn: CreateModuleFnType = create_fanet_module):
        super(PyramidDecoderModule, self).__init__()
        self.upsample = upsample
        self.block_modules = nn.ModuleList()
        self.coordinate_kwargs = {
            'num_nbrs': None,
            'pos_mlp': 64,
            'attn_mlp': 4,
            'layer_num': layer_num
        }
        if upsample:
            self.mlp = nn.Sequential(nn.ConvTranspose1d(d_in, d_out, 1, bias=True),
                                     nn.BatchNorm1d(d_out),
                                     nn.ReLU())
            d_in = d_out
        
        self.block_modules.append(
            InvertibleModuleWrapper(
                create_module_fn(
                    d_in,
                    **self.coordinate_kwargs), disable=disable_custom_gradient
                )
            )

    def forward(self, x):
        if self.upsample:
            global POOL_CNT
            POOL_CNT -= 1
            if POOL_CNT == 0: pre_x = X_DICT['101'].to(x.device)
            if POOL_CNT == 1: pre_x = X_DICT['210'].to(x.device)
            if POOL_CNT == NUM_LAYERS - 2:
                pre_x = X_DICT['x{}'.format(POOL_CNT)].to(x.device)

            # pooled_pos = sth.iter_set['xyz'][POOL_CNT]

            sth3.up_idx = sth.iter_set['interp_idx'][POOL_CNT]
            # nearest_pool = get_nearest_index(pooled_pos, pos)
            feat_map = indexing_neighbor(x.permute(0, 2, 1), sth3.up_idx).squeeze(2)
            cur_x = torch.cat([pre_x, feat_map.permute(0, 2, 1)], dim=1)
            x = self.mlp(cur_x).permute(0, 2, 1)
            # pos = pooled_pos

            # data = torch.cat([pos.permute(0, 2, 1), x], dim=1)
        sth2.n_idx = sth.iter_set['neigh_idx']  #[POOL_CNT]
            # sth2.dist = sth.iter_set['neigh_dist'][POOL_CNT]
        x = self.block_modules[0](x)  # data --> x
        return x


class DecoderModule(nn.Module):
    def __init__(self, d_in: int, d_out: int, depth: int=1, upsample: bool=True,
                 disable_custom_gradient: bool=False,  # todo: custom_gradient switch
                 create_module_fn: CreateModuleFnType = create_fanet_module):  # todo: switch btwn the different 'create *** modules'
        super(DecoderModule, self).__init__()
        self.depth = depth
        self.upsample = upsample
        self.block_modules = nn.ModuleList()
        self.coordinate_kwargs = {
            'num_nbrs': None,
            'pos_mlp': 64,
            'attn_mlp': 4,
            'layer_num': 0
        }

        if upsample:
            self.mlp = nn.Sequential(nn.ConvTranspose1d(d_in, d_out, 1, bias=False),
                                      nn.BatchNorm1d(d_out),
                                      nn.ReLU())
            d_in = d_out

        # create blocks in a layer
        for i in range(depth):
            self.block_modules.append(
                InvertibleModuleWrapper(
                    create_module_fn(
                        d_in,
                        **self.coordinate_kwargs), disable=disable_custom_gradient
                )
            )

    def forward(self, x):
        if self.upsample:
            global POOL_CNT
            POOL_CNT -= 1
            if POOL_CNT == 0: pre_x = X_DICT['x'].to(x.device)
            if POOL_CNT > 0 and POOL_CNT < NUM_LAYERS - 1:
                pre_x = X_DICT['x{}'.format(POOL_CNT)].to(x.device)

            # if POOL_CNT == 0: pooled_pos = POS_DICT['pos'].to(x.device)
            # if POOL_CNT > 0 and POOL_CNT < NUM_LAYERS - 1:
            #     # pooled_pos = POS_DICT['pos{}'.format(POOL_CNT)].to(x.device)
            # pooled_pos = sth.iter_set['xyz'][POOL_CNT]

            sth3.up_idx = sth.iter_set['interp_idx'][POOL_CNT]
            # nearest_pool = get_nearest_index(pooled_pos, pos)
            feat_map = indexing_neighbor(x.permute(0, 2, 1), sth3.up_idx).squeeze(2)
            cur_x = torch.cat([pre_x, feat_map.permute(0, 2, 1)], dim=1)
            x = self.mlp(cur_x).permute(0, 2, 1)
            # pos = pooled_pos
        # else:
        #     x = x.permute(0, 2, 1)

        for i in range(self.depth):
            # if i > 0:
            #     if POOL_CNT == 0:
            #         pos = POS_DICT['pos'].to(x.device)
            #     elif POOL_CNT > 0:
            #         pos = POS_DICT['pos{}'.format(POOL_CNT)].to(x.device)

            # data = list([pos, x])
            # data = torch.cat([pos.permute(0, 2, 1), x], dim=1)
            sth2.n_idx = sth.iter_set['neigh_idx'] #[POOL_CNT]
            # sth2.dist = sth.iter_set['neigh_dist'][POOL_CNT]
            x = self.block_modules[i](x)  # pos, x
        return x


class point_embedding(nn.Module):
    def __init__(self, hidden_dim, use_color, device):
        super(point_embedding, self).__init__()

        self.device = device
        self.use_color = use_color
        if use_color:
            self.mlp1 = nn.Sequential(
                nn.Linear(10+3, hidden_dim),  # no 4
                nn.ReLU())  # no 4; dim
        else:
            self.mlp1 = nn.Sequential(
                nn.Linear(10, hidden_dim),  # no 4
                nn.ReLU())  # no 4; dim

    def forward(self, pos):
        # if self.use_color:
        xyz = pos[:, :3, :]
        
        idx = sth.iter_set['neigh_idx'][0]
        dist = sth.iter_set['neigh_dist'][0]
        B, N, K = idx.shape

        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K) # create axis, repeat [N,K] content on new axis
        extended_xyz = xyz.unsqueeze(-1).expand(B, 3, N, K)  # .expand: repeatition along newly created axis
        neighbors = torch.gather(extended_xyz, 2, extended_idx)  # shape (B, 3, N, K)
        
        # relative point position encoding
        concat = torch.cat((extended_xyz, neighbors, extended_xyz - neighbors,
                            dist.unsqueeze(-3)), dim=-3).type(torch.FloatTensor).to(xyz.device)
        
        if self.use_color:
            pnt_emb = self.mlp1(torch.cat((torch.max(concat, dim=-1)[0], pos[:, 3:, :]), dim=1).permute(0, 2, 1))  # B N C
        else:
            pnt_emb = self.mlp1(torch.max(concat, dim=-1)[0].permute(0, 2, 1))

        return pnt_emb


class iNet(nn.Module):
    def __init__(self, num_layers: int, use_color: bool, mode: str):
        super(iNet, self).__init__()
        self.num_layers = num_layers
        self.decDepth = 1
        self.mode = mode

        self.pnt_embedding = point_embedding(hidden_dim=128, use_color=use_color, device='cuda:0')
        # self.pre_conv = nn.Sequential(
        #     nn.Conv1d(6, CHANNELS[0], 1, bias=False),
        #     nn.BatchNorm1d(CHANNELS[0]),
        #     nn.ReLU(inplace=True))
        self.mid_conv = nn.Conv1d(CHANNELS[-1], CHANNELS[-1], 1, bias=True)
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(128, cfg.num_classes, kernel_size=1, bias=True),  # Config.num_classes
        )

        # create encoder layers
        self.encoder_modules = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                downsample = False
            else:
                downsample = True
            self.encoder_modules.append(
                EncoderModule(CHANNELS[i], CHANNELS[i+1], i,
                    # getChannelsAtIndex(i-1), getChannelsAtIndex(i),
                    ENCODER_DEPTH[i], downsample=downsample
                )
            )

        # create decoder layers
        if self.mode == 'pyramid':
            self.decoder_modules = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    upsample = False
                    d_in = getChannelsAtIndex(i+1)
                else:
                    upsample = True
                    # d_in = getChannelsAtIndex(i) * 1.5
                    d_in = CHANNELS[i+1] + 128*i
                ln = i-1
                if ln < 0: ln = 0
                self.decoder_modules.append(
                    PyramidDecoderModule(
                        int(d_in),
                        128*i,  # CHANNELS[i],
                        ln,
                        upsample=upsample
                    )
                )
            self.level210Blk = PyramidLevel1Block(
                d_in = CHANNELS[2] + CHANNELS[3],
                d_out = CHANNELS[2],
                layer_num=1,
                save_var='210'
            )
            self.level101Blk = PyramidLevel1Block(
                d_in = CHANNELS[1] + CHANNELS[2],
                d_out = CHANNELS[1],
                layer_num=0,
                save_var='101'
            )
        else:
            self.decoder_modules = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    upsample = False
                    d_in = getChannelsAtIndex(i+1)
                else:
                    upsample = True
                    # d_in = getChannelsAtIndex(i) * 1.5
                    d_in = CHANNELS[i+1] + 128*i #* 1.5
                self.decoder_modules.append(
                    DecoderModule(
                        int(d_in), 
                        128*i, # CHANNELS[i], # getChannelsAtIndex(i-1),
                        DECODER_DEPTH[i], upsample=upsample
                    )
                )

    def forward(self, x):  # pos, x

        # pos = x.clone()
        # POS_DICT['pos'] = pos.clone().cpu()
        x = x.permute(0, 2, 1)

        # x = self.pre_conv(x)
        x = self.pnt_embedding(x)

        for i in range(self.num_layers):
            # pos = sth.iter_set['xyz'][i]  # .to(x.device)  # todo: remove to device
            # pos = getPosArgAtIndex(i-1).to(x.device)
            # print('\nEncoder layer {} \t'.format(i))
            x = self.encoder_modules[i](x)

        # print('mid_conv: ', end='\t')
        x = self.mid_conv(x)
        # print(x.shape)

        for i in range(self.num_layers-1, -1, -1):
            # pos = sth.iter_set['xyz'][i]
            # pos = getPosArgAtIndex(i).to(x.device)

            if i == 0:
                x = x.permute(0, 2, 1)

            # print('\n Decoder layer {} \t'.format(i))
            x = self.decoder_modules[i](x)

            if (i == 3) and (self.mode == 'pyramid'):
                # TODO: pyramid models worked and stored here
                # print('\n 210 layer')
                self.level210Blk(x)

                # print('\n 101 layer')
                self.level101Blk(x)

                global POOL_CNT
                POOL_CNT = i-1

       # print('fc_layer')
        x = self.fc_layer(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # [b n c]
        return x


import os, torch, logging
from pathlib import Path
from helper_files.tools import Config as cfg
from randla.data_prep1 import CloudsDataset, ActiveLearningSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from helper_files.tools import start_logger
from modules.net_modules70 import iNet, sth
from helper_files.data_utils import build_input1, unpack_input1, store_results


gpu = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

ROOT_PATH = (Path(cfg.root)).resolve()
data_path = ROOT_PATH / 'subsampled'  # ROOT_PATH / 'raw/train'

dataset_sampling = 'active_learning'
batch_size = 32
use_color = True
file_choice = 'L002'

seg_label_to_cat = {}
for i, cat in enumerate(cfg.class2label.keys()):
    seg_label_to_cat[i] = cat

# load data
dataset = CloudsDataset(data_path, file_choice, use_color=use_color)
test_sampler = ActiveLearningSampler(dataset, batch_size=batch_size,
                                    split='validation', use_color=use_color)
test_loader = DataLoader(test_sampler, batch_size)

d_in = next(iter(test_loader))[0].size(-1)  # returns input dim

max_epoch = 500

n_points = test_loader.dataset.dataset.pc_size
xyz_probs = np.zeros((n_points, cfg.num_classes))
xyz_probs[:] = np.nan
visited = np.zeros((n_points,), dtype=np.int32)

model = iNet(num_layers=4, use_color=use_color, mode='pyramid') # pyramid or unet
model.cuda()
if cfg.pretrain is not None:
    model.load_state_dict(torch.load(cfg.pretrain))
    print('load models %s' % cfg.pretrain)

# pretrain = cfg.pretrain
# init_epoch = int(cfg.pretrain[-21:-18]) if cfg.pretrain is not None else 0

model.eval()
# test_smooth = 0.98
n_votes = 1
with torch.no_grad():
    for step in range(max_epoch):
        print(f"Round {step}")
        for data in tqdm(test_loader, desc="segmentation", total=len(test_loader)*batch_size):
            # start_time = time.time()
            inputs = build_input1(data, cfg.num_layers)
            # print(time.time() - start_time)
            input_list = unpack_input1(inputs, cfg.num_layers, device='cuda:0')
            # print(time.time() - start_time)
            sth.iter_set = input_list
            del input_list

            model.train()
        
            pred = model(sth.iter_set['features'][:,:,:6].float().cuda())

            q_idx = data[2].cuda()

            for j in range(pred.shape[0]):
                probs = pred[j, :, :].cpu().detach().float().numpy()
                # probs = np.swapaxes(np.squeeze(probs), 0, 1)
                # ids = inputs['input_inds'][j, :].cpu().detach().int().numpy()
                ids = q_idx[j, :].cpu().detach().int().numpy()
                xyz_probs[ids] = np.nanmean([xyz_probs[ids], np.exp(probs)], axis=0)
                # xyz_probs[ids] = test_smooth * xyz_probs[ids] + (1 - test_smooth) * probs
                visited[ids] += 1
        least_visited = np.min(np.unique(visited))
        if least_visited >= n_votes:
            print(f"Each point was visited at least {n_votes}")
            break
        else:
            print(np.unique(visited, return_counts=True))

# proj_probs_list = []
num_val = len(test_loader.dataset.dataset.input_labels['validation'])
for i_val in range(num_val):
    # Reproject probs back to the evaluations points
    proj_idx = test_loader.dataset.dataset.val_proj[i_val]
    val_lbs = test_loader.dataset.dataset.val_labels[i_val]
    probs = xyz_probs[proj_idx, :]
    # proj_probs_list += [probs]

xyz_tile = test_loader.dataset.dataset.input_trees['validation'][0].data
if use_color:
    true_rgb = test_loader.dataset.dataset.colors['validation'][0].data*255.0
else:
    true_rgb = None
gt_labels = test_loader.dataset.dataset.input_labels['validation'][0].data
xyz_labels = np.argmax(xyz_probs, axis=1)

if gt_labels is not None:
    from helper_files.metrics import accuracy, intersection_over_union
    import pandas as pd

    val_ious = []
    val_accs = []

    table = pd.DataFrame(columns=['classes', 'iou_val', 'acc_val'])
    table['classes'] = [cat_value for cat_value in seg_label_to_cat.values()]
    meanAttr = [{'classes': 'OA/mIoU', 'iou_val': 'nan', 'acc_val': 'nan'}]
    table.loc[8] = list(meanAttr[0].values())

    val_accs = accuracy(torch.from_numpy(probs).unsqueeze(0).permute(0, 2, 1), torch.from_numpy(np.array(val_lbs)).long().unsqueeze(0))  # gt_labels --> val_lbs
    val_ious = intersection_over_union(torch.from_numpy(probs).unsqueeze(0).permute(0, 2, 1), torch.from_numpy(np.array(val_lbs)).long().unsqueeze(0))
    print(intersection_over_union(torch.from_numpy(xyz_probs).unsqueeze(0).permute(0, 2, 1), torch.from_numpy(np.array(gt_labels)).long().unsqueeze(0)))

    # val_accs = np.nanmean(np.array(val_accs), axis=0)
    # val_ious = np.nanmean(np.array(val_ious), axis=0)

    table['iou_val'] = [f'{iou:.3f}' if not np.isnan(iou) else ' nan' for iou in val_ious]
    table['acc_val'] = [f'{acc:.3f}' if not np.isnan(acc) else ' nan' for acc in val_accs]

    # print(table)
    start_logger(cfg)
    logging.info('batch_size: {}'.format(batch_size))
    logging.info('nvotes: {}'.format(n_votes))
    logging.info(table)

model_path = cfg.pretrain.rsplit('/', 2)[0]
results_path = store_results(model_path, xyz_tile, xyz_labels, xyz_probs, true_rgb=true_rgb,
                             gt_labels=gt_labels, pc_path = None, segmentation_name=file_choice)
# mask_map = {}
# for label in mapping.values():
#     mask = xyz_labels == label
#     mask_map[label] = mask
#
# plot = generate_k3d_plot(xyz_tile, mask_map=mask_map, mask_color=color_mapping, name_map=name_mapping)
# snapshot = plot.get_snapshot(9)
# snap_path = f"{results_path}snapshot_predictions.html"
# with open(snap_path, 'w') as fp:
#     fp.write(snapshot)
#     print(f"Labelled snapshot save at {snap_path}")


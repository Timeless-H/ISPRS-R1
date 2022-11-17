import torch 
import os, random, logging
from pathlib import Path
from helper_files.tools import Config as cfg
from helper_files.tools import start_logger
from randla.data_prep import data_loaders
from modules.net_modules70 import iNet, sth
from helper_files.data_utils import build_input1, unpack_input1, test_seg_alt
from tqdm import tqdm
import numpy as np
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from helper_files.metrics import *

def logMemoryUsage(additional_string="[*]"):
    # device = 'cuda:{}'.format(gpu)
    if torch.cuda.is_available():
        logging.info(additional_string + "Memory {:.0f}MiB max, {:.0f}MiB current".format(
            torch.cuda.max_memory_allocated()/1024/1024, torch.cuda.memory_allocated()/1024/1024
        ))

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

ROOT_PATH = (Path(cfg.root)).resolve() # already at subsampled
# data_path = ROOT_PATH / 'subsampled'  

dataset_sampling = 'active_learning'
batch_size = 16
use_color = True

'''LOG'''
start_logger(cfg)
logging.info('point set size: {}'.format(cfg.num_points))

seg_label_to_cat = {}
for i, cat in enumerate(cfg.class2label.keys()):
    seg_label_to_cat[i] = cat

tr_loader, val_loader = data_loaders(ROOT_PATH, cfg.sampling_type,
                                        batch_size=batch_size,
                                        use_color=use_color,
                                        num_workers=6,
                                        pin_memory=False)

d_in = next(iter(tr_loader))[0].size(-1)  # returns input dim

model = iNet(num_layers=4, use_color=use_color, mode='pyramid')  # pyramid or unet
paramCount = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("Parameters: {:,}".format(paramCount).replace(",", "'"))
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                             eps=1e-08, weight_decay=cfg.decay_rate)  # cfg.learning_rate

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=2, verbose=True)

criterion = torch.nn.NLLLoss()  # weight=weights

"""train net"""
best_acc = 0
meaniou = 0
best_meaniou = 0
init_epoch = 0
max_epoch = 60
losses = []
# TODO: optimize the ff with argparse

tb = SummaryWriter(comment=f'subs_type=mine | pyramid | 4heads | utm_offset | kvq-mha')

tr_accuracies = []
tr_ious = []

for epoch in range(init_epoch, max_epoch):
    tr_avgloss = 0.0
    val_avgIoU = 0.0

    for i, data in tqdm(enumerate(tr_loader, 0), total=cfg.train_steps + 1000, smoothing=0.9):
        # with torch.autograd.set_detect_anomaly(True):
        # start_time = time.time()
        inputs = build_input1(data, cfg.num_layers)
        # print(time.time() - start_time)
        input_list = unpack_input1(inputs, cfg.num_layers, device='cuda:0')
        # print(time.time() - start_time)
        sth.iter_set = input_list
        del input_list

        model.train()
        
        pred = model(sth.iter_set['features'].float()) # .cuda()

        tr_accuracies.append(accuracy(pred.permute(0, 2, 1), sth.iter_set['labels'])) # .cuda()
        tr_ious.append(intersection_over_union(pred.permute(0, 2, 1), sth.iter_set['labels'])) # .cuda()
        acc = np.nanmean(np.array(tr_accuracies), axis=0)[8]
        iou = np.nanmean(np.array(tr_ious), axis=0)[8]

        pred = pred.contiguous().view(-1, cfg.num_classes)
        target = sth.iter_set['labels'].view(-1, 1)[:, 0] # .cuda()

        loss = criterion(pred, target)

        if i < 5 and epoch < 5:
            logMemoryUsage()

        losses.append(loss.item())
        
        print('\nEpoch %d | iter %d | tr_acc: %f | tr_iou: %f | loss: %f' % (epoch, i, acc, iou, loss.item()))

        optimizer.zero_grad()
        # prof.enable()
        loss.backward()
        optimizer.step()

        # if i == 6:
        #     break
        
    sth.iter_set = []

    tr_avgloss = np.mean(losses)
    tb.add_scalar("tr_loss_per_epoch", tr_avgloss, epoch)
    logging.info('------------------------>> loss @ epoch %i: %.6f' % (epoch, tr_avgloss))

    table = pd.DataFrame(columns=['classes', 'iou_tr', 'acc_tr', 'iou_val', 'acc_val'])
    table['classes'] = [cat_value for cat_value in seg_label_to_cat.values()]
    meanAttr = [{'classes': 'OA/mIoU', 'iou_tr': 'nan', 'acc_tr': 'nan', 'iou_val': 'nan', 'acc_val': 'nan'}]
    table.loc[8] = list(meanAttr[0].values())
    accs = np.nanmean(np.array(tr_accuracies), axis=0)
    ious = np.nanmean(np.array(tr_ious), axis=0)
    table['iou_tr'] = [f'{iou:.3f}' if not np.isnan(iou) else ' nan' for iou in ious]
    table['acc_tr'] = [f'{acc:.3f}' if not np.isnan(acc) else ' nan' for acc in accs]

    tb.add_scalar("lr_per_epoch", optimizer.param_groups[0]['lr'], epoch)

    val_accs, val_ious = test_seg_alt(model, val_loader, cfg.num_classes)
    table['iou_val'] = [f'{iou:.3f}' if not np.isnan(iou) else ' nan' for iou in val_ious]
    table['acc_val'] = [f'{acc:.3f}' if not np.isnan(acc) else ' nan' for acc in val_accs]

    mean_iou = float(table['iou_val'][8])
    mean_acc = float(table['acc_val'][8])

    scheduler.step(mean_iou)

    tb.add_scalar("val_iou_per_epoch", mean_iou, epoch)

    if mean_iou > best_meaniou or epoch == max_epoch-1:  
        best_acc = mean_acc
        torch.save(model.state_dict(),
                    '%s/iNet_%.3d_%.4f_%.4f.pth' % (cfg.checkpoints_dir, epoch, mean_iou, best_acc))

        logging.info(table)
        # logging.info('Save models..')

        best_meaniou = mean_iou

    logging.info('Best accuracy is: %.5f' % best_acc)
    logging.info('Best meanIOU is: %.5f' % best_meaniou) 

print("done !")
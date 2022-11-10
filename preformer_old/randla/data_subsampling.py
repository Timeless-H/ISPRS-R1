import numpy as np
import sys
from pathlib import Path
sys.path.append('./')
from sklearn.neighbors import KDTree
import pickle

from preformer.helper_files.tools import Config as cfg
from preformer.randla.utils import DataProcessing as DP
# import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

ROOT_PATH = (Path("/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/perpetual/DATA/Toronto_ranla")).resolve()
NEW_PATH = ROOT_PATH / 'subsampled'
RAW_PATH = ROOT_PATH / 'raw'

TRAIN_PATH = 'train'
VAL_PATH = 'val'
TEST_PATH = 'test'

LABELS_AVAILABLE_IN_TEST_SET = True
use_color = True
sub_grid_size = cfg.sub_grid_size

for folder in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    (NEW_PATH / folder).mkdir(parents=True, exist_ok=True)

    for file in (RAW_PATH / folder).glob('*.npy'):
        file_name = file.stem
        print(file.name, end=':\t')
        if (NEW_PATH / folder / (file_name + '.npy')).exists():
            print('Already subsampled.')
            continue

        data = np.load(file, mmap_mode='r')
        print('Loaded data of shape: ', np.shape(data))

        # For each point cloud, a sub sample of point will be used for the nearest neighbors and training
        sub_npy_file = NEW_PATH / folder / (file_name + '.npy')
        xyz = data[:, :3].astype(np.float32)
        if use_color:
            colors = data[:, 3:6].astype(np.uint8)

            if folder != TEST_PATH or LABELS_AVAILABLE_IN_TEST_SET:
                labels = data[:, -1].astype(np.uint8)
                sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
                sub_colors = sub_colors / 255.0
                np.save(sub_npy_file, np.concatenate((sub_xyz, sub_colors, sub_labels), axis=1))
            else:
                sub_xyz, sub_colors = DP.grid_sub_sampling(xyz, colors, None, sub_grid_size)
                sub_colors = sub_colors / 255.0
                np.save(sub_npy_file, np.concatenate((sub_xyz, sub_colors), axis=1))
        else:
            if folder != TEST_PATH or LABELS_AVAILABLE_IN_TEST_SET:
                labels = data[:, -1].astype(np.uint8)
                sub_xyz, sub_labels = DP.grid_sub_sampling(xyz, None, labels, sub_grid_size)
                np.save(sub_npy_file, np.concatenate((sub_xyz, sub_labels), axis=1))
            else:
                sub_xyz = DP.grid_sub_sampling(xyz, grid_size=sub_grid_size)
                np.save(sub_npy_file, sub_xyz)
        
        # The search tree is the KD_tree saved for each point cloud
        search_tree = KDTree(sub_xyz)
        kd_tree_file = NEW_PATH / folder / (file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

            # Projection is the nearest points (in the selected grid) to each point of the cloud
            proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            proj_save = NEW_PATH / folder / (file_name + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)            


print("Amen")


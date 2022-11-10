import sys
sys.path.append('./')
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
import numpy as np
from pathlib import Path
import time, random
import pickle
from helper_files.tools import Config as cfg
from randla.utils import DataProcessing as DP
from tqdm import tqdm


ROOT_PATH = (Path("/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/perpetual/DATA/Toronto_ranla")).resolve()
TRAIN_PATH = ROOT_PATH / 'subsampled/train'


class PointCloudsDataset(Dataset):
    """this code when run in a for loop will treat each .npy file as it will treat a batch
        i.e., each .npy file per loop"""
    def __init__(self, dir, labels_available=True):
        self.paths = list(dir.glob(f'*.npy'))
        self.labels_available = labels_available

    def __getitem__(self, idx):
        path = self.paths[idx]
        points, labels = self.load_npy(path)

        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()

        return points_tensor, labels_tensor

    def __len__(self):
        return len(self.paths)

    def load_npy(self, path):
        r"""
            load the point cloud and labels of the npy file located in path

            Args:
                path: str
                    path of the point cloud
                keep_zeros: bool (optional)
                    keep unclassified points
        """
        cloud_npy = np.load(path, mmap_mode='r')
        points = cloud_npy[:,:-1] if self.labels_available else cloud_npy

        if self.labels_available:
            labels = cloud_npy[:,-1]

            # balance training set
            points_list, labels_list = [], []
            for i in range(len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels == i]), 8000)
                    points_list.append(points[labels == i][idx])
                    labels_list.append(labels[labels == i][idx])
                except ValueError:
                    continue
            if points_list:
                points = np.stack(points_list)
                labels = np.stack(labels_list)
                labeled = labels > 0
                points = points[labeled]
                labels = labels[labeled]

        return points, labels


class CloudsDataset(Dataset):
    def __init__(self, dir, data_type='npy', use_color=False):
        self.path = dir
        self.paths = list(dir.rglob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.data_type = data_type
        self.use_color = use_color
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_proj = []
        self.val_labels = []
        self.val_split = 'val'
        self.test_split = 'test'

        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            if self.val_split in str(file_path):
                # cloud_split = 'validation'
                continue
            elif self.test_split in str(file_path):
                # continue
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = file_path.parent / '{:s}_KDTree.pkl'.format(cloud_name)
            sub_npy_file = file_path.parent / '{:s}.npy'.format(cloud_name)

            data = np.load(sub_npy_file, mmap_mode='r')

            sub_colors = data[:,3:6] if self.use_color else None  # todo: if statement for 'use_color'
            sub_labels = data[:,-1].copy()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[cloud_split].append(search_tree)
            self.input_colors[cloud_split].append(sub_colors)
            self.input_labels[cloud_split].append(sub_labels)
            self.input_names[cloud_split].append(cloud_name)

            size = data.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.name, size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.paths):  # todo: work on the validation aspect
            t0 = time.time()
            cloud_name = file_path.stem

            # Validation projection and labels
            if self.val_split in str(file_path):
                proj_file = file_path.parent / '{:s}_proj.pkl'.format(cloud_name)
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
    
    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self.size  # num of clouds. as in files not points


class ActiveLearningSampler(IterableDataset):
    """ docstring """
    def __init__(self, dataset, batch_size=6, split='training', use_color=False):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.use_color = use_color
        self.possibility = {}  # ??
        self.min_possibility = {}  # ??

        if split == 'training':
            self.n_samples = cfg.train_steps  # number of steps per epoch
        else:
            self.n_samples = cfg.val_steps

        # Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        """possibilty = random values the len of sub_colors * 0.001 per sub_file
           min_possibility = the min of possibility per sub_file"""
        # for i, tree in enumerate(self.dataset.input_labels[split]):
        #     self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
        #     self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        class_list = np.arange(cfg.num_classes)
        for i in range(int(self.n_samples * self.batch_size * 1.5)):
            ans = random.choice(class_list)
            self.possibility[split].append(ans)

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        """ Choosing the least known point as center of a new cloud each time. """
        
        for i in range(self.n_samples * self.batch_size):  # num per epoch
            """ batch size here is in files, not points. 
                n_samples = num of train steps.
                so, 6 files per batch, 20 steps per file = 120 per epoch """
            
            if cfg.sampling_type == 'active_learning':
                """randomly choose a cloud, randomly choose a point from this cloud
                   to be the central point. add some noise to the central point and
                   find its k=40960 nearest neighbors. shuffle the obtained knn indexes n
                   get the shuffled xyz, color & labels using the indexes. Determine 
                   relative neighborhood point positions, delta_dists.
                   update the (min) possibility folders.
                   if points in file are less than k=40960, apply augmentaion.
                   move from numpy array to torch tensor."""
                # choose a random cloud
                # cloud_idx = int(np.argmin(self.min_possibility[self.split]))
                cloud_idx = int(random.choice(np.arange(len(self.dataset.input_names[self.split]))))


                # choose the point with the minimum of possibility as query point
                # point_ind = np.argmin(self.possibility[self.split][cloud_idx])
                class_possibility = self.possibility[self.split].pop()
                temp = np.array(self.dataset.input_labels[self.split][cloud_idx].data)
                possibility_set = np.where(temp == class_possibility)
                cnt = 0
                while len(possibility_set[0]) == 0:
                    cloud_idx = int(random.choice(np.arange(len(self.dataset.input_names[self.split]))))
                    temp = np.array(self.dataset.input_labels[self.split][cloud_idx].data)
                    possibility_set = np.where(temp == class_possibility)
                    cnt += 1
                    if cnt == len(self.dataset.input_names[self.split]) * 2:
                        class_possibility = self.possibility[self.split].pop()
                del cnt
                possibility_set = list(possibility_set[0])
                # random.shuffle(possibility_set)
                # point_ind = possibility_set[0]
                point_ind = possibility_set[random.choice(np.arange(len(possibility_set)))]

                # Get points from tree structure
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                
                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                # noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point # + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx] if self.use_color else None  # todo: add 'use_color' if statement
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]
                # print(np.unique(self.dataset.input_labels[self.split][cloud_idx].data))

                # dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                # delta = np.square(1 - dists / np.max(dists))
                # self.possibility[self.split][cloud_idx][queried_idx] += delta
                # self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))
                self.possibility[self.split].insert(len(self.possibility[self.split]), random.choice(np.arange(cfg.num_classes)))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            # Simple random choice of cloud and points in it
            elif cfg.sampling_type=='random':

                cloud_idx = np.random.choice(len(self.min_possibility[self.split]), 1)[0]
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                queried_idx = np.random.choice(len(self.dataset.input_trees[self.split][cloud_idx].data), cfg.num_points)
                queried_pc_xyz = points[queried_idx]
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float() if self.use_color else None  # todo: add 'use_color' if statement
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

            points = torch.cat((queried_pc_xyz, queried_pc_colors), 1) if self.use_color else queried_pc_xyz  # todo: add 'use_color' if statement

            yield [points, queried_pc_labels, queried_idx, cloud_idx]


def data_loaders(dir, sampling_method='active_learning', **kwargs):
    if sampling_method == 'active_learning':
        use_color = kwargs.get('use_color')
        dataset = CloudsDataset(dir, use_color=use_color)
        batch_size = kwargs.get('batch_size', 6)
        val_sampler = ActiveLearningSampler(dataset, batch_size=batch_size,
                                            split='validation', use_color=use_color)
        train_sampler = ActiveLearningSampler(dataset, batch_size=batch_size,
                                              split='training', use_color=use_color)
        del kwargs['use_color']
        return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)

    if sampling_method == 'naive':
        train_dataset = PointCloudsDataset(dir) # / train
        # val_dataset = PointCloudsDataset(dir) # / val
        return DataLoader(train_dataset, shuffle=True, **kwargs)  # , DataLoader(val_dataset, **kwargs)

    raise ValueError(f"Dataset sampling method '{sampling_method}' does not exist.")


if __name__ == '__main__':
    
    tr_dataset = CloudsDataset(TRAIN_PATH)
    batch_sampler = ActiveLearningSampler(tr_dataset, batch_size=4)
    tr_loader = DataLoader(batch_sampler)
    for data in tr_loader:
        data, labels, query_idx, cloud_idx = data
        print('Number of points:', len(data))
        print('Point position:', data[0, 1, :3])
        print('Color:', data[0, 1, 3:])
        print('Label:', labels[0, 1])
        print('Index of point:', query_idx[0, 1])
        print('Cloud index:', cloud_idx)
    #     break
    pass
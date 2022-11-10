import numpy as np
# from randla.cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling


class DataProcessing:

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    # @staticmethod
    # def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    #     """
    #     CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    #     :param points: (N, 3) matrix of input points
    #     :param features: optional (N, d) matrix of features (floating number)
    #     :param labels: optional (N,) matrix of integer labels
    #     :param grid_size: parameter defining the size of grid voxels
    #     :param verbose: 1 to display
    #     :return: sub_sampled points, with features and/or labels depending of the input
    #     """

    #     if (features is None) and (labels is None):
    #         return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
    #     elif labels is None:
    #         return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
    #     elif features is None:
    #         return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    #     else:
    #         return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
    #                                        verbose=verbose)

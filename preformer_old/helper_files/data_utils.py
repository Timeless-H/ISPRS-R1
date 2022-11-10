import torch, sys, os
sys.path.append('./')

import numpy as np
from knn_cuda import KNN
from preformer.helper_files.tools import Config as cfg
from datetime import datetime
import pickle


def build_input1(data, num_layers):
    seed_value = 42
    xyz, labels, _, _ = data
    coords = xyz[:, :, :3].cuda()
    labels = torch.tensor(labels, dtype=torch.long)
    # query_idx = torch.tensor(query_idx, dtype=torch.int32)
    # pc_id = torch.tensor(pc_id, dtype=torch.int32)
    input_points = []
    input_neighbors = []
    input_distance = []
    input_pools = []
    input_up_samples = []

    # knn = KNN(k=16, transpose_mode=True)
    for i in range(num_layers):
        torch.manual_seed(seed_value)
        knn = KNN(k=cfg.k_nbrs, transpose_mode=True)
        neigh_dist, neigh_idx = knn(coords, coords)
        # _, neigh_idx = cKDTree(coords.reshape(-1, 3)).query(coords, k=k_nbrs)
        sub_sampling_idx = coords.shape[1]//cfg.sub_sampling_ratio[i]
        sub_points = coords[:, :sub_sampling_idx, :] # todo: try one for fps
        pool_i = neigh_idx[:, :sub_sampling_idx, :]

        knn = KNN(k=1, transpose_mode=True)
        _, up_i = knn(sub_points, coords)
        # _, up_i = cKDTree(sub_points[:, :, :3].reshape(-1, 3)).query(coords, k=1)
        input_points.append(torch.tensor(coords, dtype=torch.float32))
        input_neighbors.append(torch.tensor(neigh_idx, dtype=torch.long))
        input_distance.append(torch.tensor(neigh_dist, dtype=torch.float32))
        input_pools.append(torch.tensor(pool_i, dtype=torch.int32))
        input_up_samples.append(torch.tensor(up_i, dtype=torch.int32))
        coords = sub_points

    inputs = input_points + input_neighbors + input_distance + input_pools + input_up_samples
    inputs += [xyz, labels]
    return inputs


def unpack_input1(input_list, n_layers, device):
    inputs = dict()
    inputs['xyz'] = input_list[:n_layers]
    inputs['neigh_idx'] = input_list[n_layers: 2 * n_layers]
    inputs['neigh_dist'] = input_list[2 * n_layers:3 * n_layers]
    inputs['sub_idx'] = input_list[3 * n_layers:4 * n_layers]
    inputs['interp_idx'] = input_list[4 * n_layers:5 * n_layers]
    # for key, val in inputs.items():
    #     inputs[key] = [x.to(device) for x in val]
    inputs['features'] = input_list[5 * n_layers]  #.to(device)
    inputs['labels'] = input_list[5 * n_layers + 1]  #.to(device)
    # inputs['input_inds'] = input_list[5 * n_layers + 2].to(device)
    # inputs['cloud_inds'] = input_list[5 * n_layers + 3].to(device)
    return inputs


def build_input_test(xyz, labels, num_layers):
    seed_value = 42
    coords = xyz[:, :, :3].cuda()
    labels = torch.tensor(labels, dtype=torch.long)
    # query_idx = torch.tensor(query_idx, dtype=torch.int32)
    # pc_id = torch.tensor(pc_id, dtype=torch.int32)
    input_points = []
    input_neighbors = []
    input_distance = []
    input_pools = []
    input_up_samples = []

    for i in range(num_layers):
        torch.manual_seed(seed_value)
        knn = KNN(k=cfg.k_nbrs, transpose_mode=True)
        neigh_dist, neigh_idx = knn(coords, coords)
        # _, neigh_idx = cKDTree(coords.reshape(-1, 3)).query(coords, k=k_nbrs)
        sub_sampling_idx = coords.shape[1]//cfg.sub_sampling_ratio[i]
        sub_points = coords[:, :sub_sampling_idx, :] # todo: try one for fps
        pool_i = neigh_idx[:, :sub_sampling_idx, :]

        knn = KNN(k=1, transpose_mode=True)
        _, up_i = knn(sub_points, coords)
        # _, up_i = cKDTree(sub_points[:, :, :3].reshape(-1, 3)).query(coords, k=1)
        input_points.append(torch.tensor(coords, dtype=torch.float32))
        input_neighbors.append(torch.tensor(neigh_idx, dtype=torch.long))
        input_distance.append(torch.tensor(neigh_dist, dtype=torch.float32))
        input_pools.append(torch.tensor(pool_i, dtype=torch.int32))
        input_up_samples.append(torch.tensor(up_i, dtype=torch.int32))
        coords = sub_points

    inputs = input_points + input_neighbors + input_distance + input_pools + input_up_samples
    inputs += [xyz, labels]
    return inputs


def test_seg_alt(model, loader, num_classes):
    from tqdm import tqdm
    from helper_files.metrics import accuracy, intersection_over_union
    from modules.net_modules70 import sth
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    model.eval()

    val_ious = []
    val_accs = []
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(loader), total=cfg.val_steps + 180, smoothing=0.9):
            xyz, labels, _, _ = data
            inputs = build_input_test(xyz, labels, num_layers=4)
            # print(time.time() - start_time)
            input_list = unpack_input1(inputs, n_layers=4, device='cuda:0')
            # print(time.time() - start_time)
            sth.iter_set = input_list
            del input_list

            pred = model(sth.iter_set['features'][:,:,:6].float().cuda())
            # points = points.transpose(2, 1)
            target = sth.iter_set['labels'].cuda()

            val_accs.append(accuracy(pred.permute(0, 2, 1), target))
            val_ious.append(intersection_over_union(pred.permute(0, 2, 1), target))

        sth.iter_set = []
    return np.nanmean(np.array(val_accs), axis=0), np.nanmean(np.array(val_ious), axis=0)


def check_create_folder(folder_path):
    if not os.path.exists(os.path.dirname(folder_path)):
        try:
            os.makedirs(os.path.dirname(folder_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise


def store_results(model_path, xyz_tile, xyz_labels, xyz_probs, true_rgb,
                  gt_labels, pc_path, segmentation_name):
    """
        Stores segmentation results that will be used to generate analysis
        and to upload data to ISIN db
    Args:
        model_path: path to the model used for segmentation
        xyz_tile: [x,y,z] array of points
        xyz_labels: array of labels for each point
        xyz_probs: array of model outputs for each point
        true_rgb: [r,g,b] array for each point
        gt_labels: ground truth labels for each point
        pc_path: path containing segmented pc file
            segmented pc
        segmentation_name: name for the segmentation folder. If None, timestamp
            will be used
    """
    print("Storing segmentation results")
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if segmentation_name is None:
        segmentation_name = date

    results_path = f"{model_path}/output/segmentations/{segmentation_name}/"
    check_create_folder(results_path)

    with open(f"{results_path}/xyz_tile.pickle", "wb") as pickle_out:
        pickle.dump(np.array(xyz_tile), pickle_out)

    with open(f"{results_path}/xyz_probs.pickle", "wb") as pickle_out:
        pickle.dump(xyz_probs, pickle_out)

    with open(f"{results_path}/xyz_labels.pickle", "wb") as pickle_out:
        pickle.dump(xyz_labels, pickle_out)

    if true_rgb is not None:
        with open(f"{results_path}/true_rgb.pickle", "wb") as pickle_out:
            pickle.dump(true_rgb, pickle_out)

    if gt_labels is not None:
        with open(f"{results_path}/gt_labels.pickle", "wb") as pickle_out:
            pickle.dump(np.array(gt_labels), pickle_out)

    # metadata = read_metadata(pc_path)
    # metadata['timestamp'] = date
    # create_metadata(results_path, **metadata)
    np.savetxt(f"{results_path}/{segmentation_name}.txt", np.column_stack([np.array(xyz_tile), xyz_labels]), fmt='%.5f %.5f %.5f %i')
    np.savetxt(f"{results_path}/{segmentation_name}gt.txt", np.column_stack([np.array(xyz_tile), np.array(gt_labels)]), fmt='%.5f %.5f %.5f %i')

    print(f"Results stored at: {results_path}")
    return results_path
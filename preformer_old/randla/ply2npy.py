from pathlib import Path
from tqdm import tqdm
import numpy as np
from ply import read_ply


ROOT_PATH = (Path("/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/perpetual/DATA/Toronto_ranla")).resolve()
# DATA_PATH = ROOT_PATH / 'subsampled'
RAW_PATH = ROOT_PATH / 'raw'
TRAIN_PATH = 'train'
TEST_PATH = 'test'
VAL_PATH = 'val'

# for folder in [TRAIN_PATH, TEST_PATH, VAL_PATH]:
#     if not (RAW_PATH / folder).exists():
#         (RAW_PATH / folder).mkdir(parents=True, exist_ok=False)

data_has_rgb = True
i_dont_have_a_lot_of_memory_ok = False
output_type = 'npy'

print(f'Computing point clouds as {output_type} files. This operation is very time-consuming.')
raw_ls = RAW_PATH.rglob('*.ply')

for pc_path in RAW_PATH.rglob('*.ply'):
    name = pc_path.stem
    phase = pc_path.parts[7]

    if output_type == 'ply':
        pc_name = name + '.ply'
    elif output_type == 'npy':
        pc_name = name + '.npy'
    else:
        raise 'unknown output_type'

    # Check if the point cloud has already been computed
    # note checking is done in the ROOT_PATH and all its sub_dir
    if list(RAW_PATH.rglob(pc_name)) != []:
        continue  # means files hv not been written yet so go ahead n write, otherwise skip file writing

    print(f'Writing {pc_name}...')

    # In order to avoid memory over flow
    if i_dont_have_a_lot_of_memory_ok:
        dir = TEST_PATH
        bunchsize = 1e5
        print(bunchsize)
        index = 0
        points = []

        with open(pc_path, 'r') as file:
            for line in tqdm(file):
                points = read_ply(pc_path)
                points = np.vstack((points['x'], points['y'], points['z'],
                                    points['red'], points['green'], points['blue'])).T
                points.append(np.loadtxt(line, dtype=np.float32))
                if len(points) == bunchsize:
                    points = np.array(points)
                    prev_data = np.load(dir/pc_name)
                    np.save(dir / pc_name, np.concatenate((prev_data, points)))
        """the point of this is to read the data in each line of the .txt file
            n when its up to the  bunchsize, u save it. 
            Saving to .npy from txt in batches of bunchsize"""
    else:
        try:
            points = read_ply(pc_path)
            if data_has_rgb:
                if phase == 'train' or phase == 'val' or phase == 'test':
                    if pc_path.parts[5] == 'Toronto_ranla':
                        utm_offset = np.array([-627400.00, -4842600.00, 0.00])
                        labels = points['scalar_Label']
                        points = np.vstack((points['x'], points['y'], points['z'], points['red'],
                                            points['green'], points['blue'], points['scalar_Label'])).T
                    else:
                        raise ValueError('Unknown data source specified!... please check!!')
                    print(np.unique(points[:, 6], return_counts=True))
                    indices = (labels != 0)
                    points = points[indices, :]
                    print(np.unique(points[:, 6], return_counts=True))
                    points[:, 6] = points[:, 6] - 1
                    print(np.unique(points[:, 6], return_counts=True))
                    points[:, :3] = points[:, :3] + utm_offset
                # elif phase == 'test':
                #     points = np.vstack((points['x'], points['y'], points['z'], points['red'],
                #                         points['green'], points['blue'])).T
            else:
                if phase == 'train' or phase == 'val' or phase == 'test':
                    if pc_path.parts[5] == 'NPM3D':
                        labels = points['class']
                        points = np.vstack((points['x'], points['y'], points['z'], points['class'])).T
                    elif pc_path.parts[5] == 'Toronto_ranla':
                        labels = points['scalar_Label']
                        points = np.vstack((points['x'], points['y'], points['z'], points['scalar_Label'])).T
                    else:
                        raise ValueError('Unknown data source specified!... please check!!')
                    indices = (labels != 0)
                    points = points[indices, :]
                    points[:, 3] = points[:, 3] - 1
                # elif phase == 'test':
                #     points = np.vstack((points['x'], points['y'], points['z'])).T
        except:
            continue

        dir = pc_path.parent

        np.save(dir / pc_name, points)
        print(f'As {dir / pc_name}...')

    print('Done.')

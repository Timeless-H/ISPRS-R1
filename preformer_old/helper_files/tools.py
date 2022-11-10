from os.path import join
import logging
from pathlib import Path
from datetime import datetime


class Config(object):
    data_name = 'toronto_3d'  # [toronto_3d, pl3d]
    num_points = 4096  # Number of input points
    sub_grid_size = 0.04  # preprocess_parameter

    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/' + str(datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    if data_name == 'toronto_3d':
        d_in = 6
        num_classes = 8
        # class_weights = [35503, 1500, 4626, 18234, 579, 742, 3733, 387]
        class_weights = [6353, 301, 1942, 866, 84, 155, 199, 24]
        classes = ['ground', 'road_markings', 'natural', 'building', 'utility_line', 'pole', 'car', 'fence']
        class2label = {cls: i for i, cls in enumerate(classes)}  # keys:value  category:label  str:num
        root = '/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/perpetual/DATA/Toronto_ranla'
    
    train_steps = 500  # Number of steps per epochs
    val_steps = 90    # Number of validation steps per epoch

    sampling_type = 'active_learning'

    optimizer = 'Adam'
    learning_rate = 0.001
    decay_rate = 1e-5

    alpha = 0.2
    dropout = 0.5

    num_layers = 4
    k_nbrs = 16
    sub_sampling_ratio = [4, 4, 4, 4]
    layer_num = []

    # pretrain = None
    pretrain = '/home/perpetual/PycharmProjects/toronto_randla/experiments/2021-12-13_20-42/checkpoints/iNet_045_0.7360_0.8980.pth'

    attn_type = "nystrom"  # softmax / fast_attn
    num_heads = [2, 2, 2, 2]  #(512/2)/64  : input_dim/head_dim
    head_dim = [32, 64, 96, 128]  # level 3 alt: 3H, 64Hdim
    # seq_len = [4096, 1024, 256, 64]
    # num_landmarks = [64, 16, 4, 1]  # u get seq_len/num_landmks subsets... the higher this val, the larger the chunks, the smaller the num of chunks
    # conv_kernel_size = 16  # just choosing a value here, to noe its importance

    # [8 4 2 1] --> [512 256 128 64]
    # [64 16 4 1] --> [64, 64, 64, 64]


def start_logger(Config):
    """Set the logger to log info in terminal and file `log_path`.
    i.e.,'model_dir/tr_log'
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # logging to file
    file_handler = logging.FileHandler(str(Config.log_dir) + '/iNet_log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('\t\t %(message)s'))
    logger.addHandler(stream_handler)

    logging.info('Dataset name: %s' % (Config.data_name))
    logging.info('Dropout rate: %f' % (Config.dropout))
    # logging.info('#Heads: %s' % (Config.num_heads))
    # logging.info('Head_dims: %s' % (Config.head_dim))
    # logging.info('Sequence length: %s' % (Config.seq_len))
    # logging.info('#Landmarks: %s' % (Config.num_landmarks))


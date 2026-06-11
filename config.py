import argparse
import numpy as np
import random
import torch

# Process bool argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise

# Process none argument
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return v

# Define configuration class
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.datasets = ['SumMe','TVSum']
        self.SumMe_len = 25
        self.TVSum_len = 50

        # Set device
        if self.device!='cpu':
            torch.cuda.set_device(self.device)

        # Set seed
        self.set_seed()

    # Set the seed
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device!='cpu':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Define all configurations
def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True, choices=['SumMe', 'TVSum'], help='Which dataset splits to apply')
    parser.add_argument('--data_path', type=str, required=True, help='Full exact path to the .h5 data file')

    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', default='1')
    parser.add_argument('--learning_rate', default='1e-3')
    parser.add_argument('--weight_decay', default='1e-7')
    
    parser.add_argument('--model_name', type=str, default='MARs')

    # MARs specific parameters
    parser.add_argument('--conformer_model_dim', type=int, default=256)
    parser.add_argument('--conformer_nhead', type=int, default=2)
    parser.add_argument('--conformer_num_blocks', type=int, default=4)
    parser.add_argument('--conformer_conv_kernel_sizes', type=int, nargs='+', default=[3,5,7,11,31,65])
    parser.add_argument('--conformer_dropout', type=float, default=0.3)
    parser.add_argument('--conformer_order', type=str, default='att_first') 

    # Generate summary videos
    parser.add_argument('--input_is_file', type=str2bool, default='true')
    parser.add_argument('--file_path', type=str, default='./SumMe/Jumps.mp4')
    parser.add_argument('--dir_path', type=str, default='./SumMe')
    parser.add_argument('--ext', type=str, default='mp4')
    parser.add_argument('--sample_rate', type=int, default=15)
    parser.add_argument('--save_path', type=str, default='./summary_videos')
    parser.add_argument('--weight_path', type=str, default='./weights/SumMe/split4.pt')

    kwargs = vars(parser.parse_args())
    kwargs.update(optional_kwargs)

    return Config(**kwargs)

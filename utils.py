import csv
import numpy as np
import torch

from collections import Counter

from models.MARs import MARs

# Count the number of parameters
def count_parameters(model):
    x = [param.numel() for name,param in model.named_parameters() if param.requires_grad]
    return sum(x) / (1024 * 1024)

# Function printing the number of parameters of MARs model
def report_params(conformer_model_dim=256,
                  conformer_nhead=4,
                  conformer_num_blocks=4,
                  conformer_conv_kernel_sizes=[31],
                  conformer_dropout=0.1,
                  conformer_order='att_first'):
    model = MARs(
        input_dim=1024,  # GoogleNet pool5 feature dimension
        model_dim=conformer_model_dim,
        nhead=conformer_nhead,
        num_blocks=conformer_num_blocks,
        conv_kernel_sizes=conformer_conv_kernel_sizes,
        dropout=conformer_dropout,
        order=conformer_order
    )
    print(f"PARAMS: {count_parameters(model):.2f}M")

# Print all arguments and GPU setting
def print_args(args):
    print(args.kwargs)
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    if 'cuda' in args.device:
        print(f"GPU: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load ground truth for TVSum
def get_gt(dataset):
    if dataset=='TVSum':
        annot_path = f"./data/ydata-anno.tsv"
        with open(annot_path) as annot_file:
            annot = list(csv.reader(annot_file, delimiter="\t"))
        annotation_length = list(Counter(np.array(annot)[:, 0]).values())
        user_scores = []
        for idx in range(1,51):
            init = (idx - 1) * annotation_length[idx-1]
            till = idx * annotation_length[idx-1]
            user_score = []
            for row in annot[init:till]:
                curr_user_score = row[2].split(",")
                curr_user_score = np.array([float(num) for num in curr_user_score])
                curr_user_score = curr_user_score / curr_user_score.max(initial=-1)
                curr_user_score = curr_user_score[::15]

                user_score.append(curr_user_score)
            user_scores.append(user_score)
        return user_scores
    elif dataset=='SumMe':
        return None
    else:
        raise

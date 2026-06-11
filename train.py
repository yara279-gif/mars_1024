import h5py
import numpy as np
import shutil
import torch

from tqdm import tqdm

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params, print_args, get_gt

# Load configurations
config = get_config()

# Print information of setting
print_args(config)

# Print the number of parameters
report_params(
    conformer_model_dim=config.conformer_model_dim,
    conformer_nhead=config.conformer_nhead,
    conformer_num_blocks=config.conformer_num_blocks,
    conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
    conformer_dropout=config.conformer_dropout,
    conformer_order=config.conformer_order
)

# Start training
dataset = config.dataset_name
user_scores = get_gt(dataset)

if dataset=='SumMe':
    batch_size = 1 if config.batch_size=='1' else int(config.SumMe_len*0.8*float(config.batch_size))
elif dataset=='TVSum':
    batch_size = 1 if config.batch_size=='1' else int(config.TVSum_len*0.8*float(config.batch_size))

for split_id,(train_loader,test_loader) in enumerate(create_dataloader(dataset, config.data_path)):
    model = set_model(
        conformer_model_dim=config.conformer_model_dim,
        conformer_nhead=config.conformer_nhead,
        conformer_num_blocks=config.conformer_num_blocks,
        conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
        conformer_dropout=config.conformer_dropout,
        conformer_order=config.conformer_order
    )
    model.to(config.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=float(config.learning_rate),weight_decay=float(config.weight_decay))

    model_selection_kendall = -1
    model_selection_spear = -1
    matching_f1_score = -1
    best_f1_score = -1
    best_kendall_epoch = -1
    best_f1_epoch = -1

    for epoch in tqdm(range(config.epochs),desc=f'Split{split_id+1}'):
        model.train()
        update_loss = 0.0
        batch = 0

        for feature,gtscore,dataset_name,video_num in train_loader:
            feature = feature.to(config.device)
            gtscore = gtscore.to(config.device)
            output = model(feature)

            loss = criterion(output,gtscore) 
            # loss.requires_grad_(True)
            
            update_loss += loss
            batch += 1

            if batch==batch_size:
                optimizer.zero_grad()
                update_loss = update_loss / batch
                update_loss.backward()
                optimizer.step()
                update_loss = 0.0
                batch = 0

        if batch>0:
            optimizer.zero_grad()
            update_loss = update_loss / batch
            update_loss.backward()
            optimizer.step()
            update_loss = 0.0
            batch = 0

        val_spears = []
        val_kendalls = []
        val_f1s = []
        model.eval()
        with torch.no_grad():
            for feature,gtscore,dataset_name,video_num in test_loader:
                feature = feature.to(config.device)
                gtscore = gtscore.to(config.device)
                output = model(feature)

                if dataset_name in ['SumMe','TVSum']:
                    with h5py.File(config.data_path,'r') as hdf:
                        user_summary = np.array(hdf[video_num]['user_summary'])
                        sb = np.array(hdf[f"{video_num}/change_points"])
                        n_frames = np.array(hdf[f"{video_num}/n_frames"])
                        positions = np.array(hdf[f"{video_num}/picks"])
                    scores = output.squeeze().clone().detach().cpu().numpy().tolist()
                    summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
                    if dataset_name == 'SumMe':
                        spear, kendall, f1 = get_corr_coeff(
                            [summary],
                            [video_num],
                            dataset_name,
                             
                            generated_summary=summary,
                            user_summary=user_summary
                        )
                    elif dataset_name == 'TVSum':
                        spear, kendall, f1 = get_corr_coeff(
                            [scores],
                            [video_num],
                            dataset_name,
                            user_scores=user_scores,
                            generated_summary=summary,
                            user_summary=user_summary
                        )
                         
                            
                    val_spears.append(spear)
                    val_kendalls.append(kendall)
                    val_f1s.append(f1)

        epoch_kendall = np.mean(val_kendalls)
        epoch_spear = np.mean(val_spears)
        epoch_f1 = np.mean(val_f1s)

        # Best Kendall/Spearman → save weights + record matching F1
        if epoch_kendall > model_selection_kendall and epoch_spear > model_selection_spear:
            model_selection_kendall = epoch_kendall
            model_selection_spear = epoch_spear
            matching_f1_score = epoch_f1
            best_kendall_epoch = epoch + 1
            torch.save(model.state_dict(), './tmp/weight.pt')

        # Best F1 → just track, no weight saving
        if epoch_f1 > best_f1_score:
            best_f1_score = epoch_f1
            best_f1_epoch = epoch + 1

    print(f"  [Split {split_id+1}] Best Kendall/Spear → Epoch {best_kendall_epoch}: Kendall:{model_selection_kendall:.3f}, Spear:{model_selection_spear:.3f}, F1:{matching_f1_score:.3f}")
    print(f"  [Split {split_id+1}] Best F1            → Epoch {best_f1_epoch}: F1:{best_f1_score:.3f}")

    shutil.move('./tmp/weight.pt', f'./weights/{dataset}/split{split_id+1}.pt')
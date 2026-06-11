import h5py
import numpy as np
import torch

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params, get_gt
from visualization import PipelineVisualizer, _interpolate_scores
viz = PipelineVisualizer("viz_output")
# Load configurations
config = get_config()

# Print the number of parameters
report_params(
    conformer_model_dim=config.conformer_model_dim,
    conformer_nhead=config.conformer_nhead,
    conformer_num_blocks=config.conformer_num_blocks,
    conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
    conformer_dropout=config.conformer_dropout,
    conformer_order=config.conformer_order
)

# Start testing
dataset = config.dataset_name
user_scores = get_gt(dataset)
split_kendalls = []
split_spears = []
split_f1s = []  # Track F1-score for each of the 5 splits

for split_id, (train_loader, test_loader) in enumerate(create_dataloader(dataset, config.data_path)):
    model = set_model(
        conformer_model_dim=config.conformer_model_dim,
        conformer_nhead=config.conformer_nhead,
        conformer_num_blocks=config.conformer_num_blocks,
        conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
        conformer_dropout=config.conformer_dropout,
        conformer_order=config.conformer_order
    )
    model.load_state_dict(torch.load(f'./weights/{dataset}/split{split_id+1}.pt', map_location='cpu'))
    model.to(config.device)
    model.eval()

    kendalls = []
    spears = []
    f1s = []  # Track F1-score for each video inside this specific split
    
    with torch.no_grad():
        for feature, _, dataset_name, video_num in test_loader:
            feature = feature.to(config.device)
            output = model(feature)

            with h5py.File(config.data_path, 'r') as hdf:
                user_summary = np.array(hdf[video_num]['user_summary'])
                sb = np.array(hdf[f"{video_num}/change_points"])
                n_frames = np.array(hdf[f"{video_num}/n_frames"])
                positions = np.array(hdf[f"{video_num}/picks"])
                gt_sampled_scores = np.array(hdf[f"{video_num}/gtscore"])
                
            scores = output.squeeze().clone().detach().cpu().numpy().tolist()
            
            gt_dense_scores = _interpolate_scores(gt_sampled_scores, positions, n_frames)
            
            # Generates the Knapsack-based binary selection using video change points
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
            dense_scores = _interpolate_scores(np.array(scores), positions, n_frames)
            shot_scores  = [dense_scores[s[0]:s[1]+1].mean().item() for s in sb]
            selected_shots = [i for i, s in enumerate(sb)
                              if summary[s[0]:s[1]+1].any()]
            
            # GT summary — متاح في SumMe فقط
            gt_summary = None
            if dataset_name == 'SumMe':
                gt_summary = (user_summary.mean(0) > 0.5).astype(np.int8)

            # ── 02: Frame scores ──────────────────────────────────────────
            viz.plot_frame_scores(
                scores, n_frames, positions,
                shot_bound=sb, summary=summary,
                video_name=video_num
            )

            # ── 03: Shot importance ───────────────────────────────────────
            viz.plot_shot_importance(
                sb, shot_scores, selected_shots,
                video_name=video_num
            )

            # ── 04: Summary selection ─────────────────────────────────────
            viz.plot_summary_selection(
                summary, sb, dense_scores,
                gt_summary=gt_summary,
                video_name=video_num
            )

            # ── 05: Predicted vs GT (SumMe فقط عندها GT) ─────────────────
            if gt_summary is not None:
                viz.plot_predicted_vs_gt(
                    summary, gt_summary, dense_scores, gt_dense_scores,
                    video_name=video_num
                )

            # 05b - The Analytical View (NEW)
            viz.plot_sampled_scores(
                sampled_scores = np.array(output.squeeze().cpu().numpy().tolist()),       # array (n_steps,) - predicted
                gt_sampled_scores = gt_sampled_scores,    # array (n_steps,) - ground truth
                video_name = video_num,
            )

            # ── 08: Pipeline summary card ─────────────────────────────────
            viz.plot_pipeline_summary(
                video_name=video_num,
                n_frames=int(n_frames),
                n_shots=len(sb),
                n_selected_shots=len(selected_shots),
                n_summary_frames=int(summary.sum()),
            )


            # Pass the Knapsack 'summary' directly to both datasets for unified evaluation
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
                                
                            
            spears.append(spear)
            kendalls.append(kendall)
            f1s.append(f1)
            
    split_kendalls.append(np.mean(kendalls))
    split_spears.append(np.mean(spears))
    split_f1s.append(np.mean(f1s))
    
    # Updated print statement to display F1 alongside correlation values per split
    print("[Split {}] Kendall:{:.3f}, Spear:{:.3f}, F1-Score:{:.3f}".format(
        split_id + 1, split_kendalls[split_id], split_spears[split_id], split_f1s[split_id]
    ))

# Final macro-average across all 5 splits
print("[FINAL - {}] Kendall:{:.3f}, Spear:{:.3f}, F1-Score:{:.3f}".format(
    dataset, np.mean(split_kendalls), np.mean(split_spears), np.mean(split_f1s)
))
# ── 07: Evaluation results (بعد كل الـ splits) ───────────────────────────────
viz.plot_evaluation_results(
    split_kendalls, split_spears,
    dataset_name=dataset
)
print()
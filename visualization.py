"""
visualization.py — MARs Video Summarization Pipeline Visualizer
================================================================
Plots (8 total, saved with ordered prefix):
  01_feature_similarity
  02_frame_scores
  03_shot_importance
  04_summary_selection
  05_predicted_vs_gt        ← UPDATED: overlays GT scores vs Predicted scores
  05b_sampled_scores        ← NEW: Non-interpolated analytical view (Sampled frames)
  06_keyframes
  07_evaluation_results
  08_pipeline_summary

Usage (standalone):
    python visualization.py --video path/to/video.mp4 --hdf path/to/data.h5 --video_num video_1

Usage (integrated):
    from visualization import PipelineVisualizer, _interpolate_scores
    viz = PipelineVisualizer(output_dir="viz_output")
"""

import argparse
import re
from pathlib import Path

import cv2
import h5py
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

PALETTE = {
    "bg":      "#ffffff",
    "panel":   "#f7f8fa",
    "accent1": "#1a6fd4",   # blue   – selected / high score
    "accent2": "#d4761a",   # amber  – shot boundaries
    "accent3": "#17936b",   # green  – ground truth
    "accent4": "#c0392b",   # red    – dropped / low score
    "accent5": "#7b2fb5",   # purple – predicted
    "text":    "#1a1c23",
    "subtext": "#6b7280",
    "grid":    "#e2e4ea",
    "border":  "#c8cad2",
}

_CMAP = LinearSegmentedColormap.from_list(
    "mars_light",
    ["#e8edf7", "#6699dd", "#1a3f8f"],
    N=256,
)

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# --- MODIFICATION 1: Added 05b for the new analytical plot ---
_PREFIX = {
    "feature_similarity":  "01",
    "frame_scores":        "02",
    "shot_importance":     "03",
    "summary_selection":   "04",
    "predicted_vs_gt":     "05",
    "sampled_scores":      "05b", 
    "keyframes":           "06",
    "evaluation_results":  "07",
    "pipeline_summary":    "08",
    "paper_style_summary": "09"
}


def _save(fig: plt.Figure, out_dir: Path, plot_key: str, video_name: str) -> None:
    """Save with ordered prefix: 02_video_1_frame_scores.png"""
    prefix = _PREFIX.get(plot_key, "00")
    filename = f"{prefix}_{video_name}_{plot_key}.png"
    path = str(out_dir / filename)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    print(f"  [viz] saved → {path}")


def _interpolate_scores(scores, positions, n_frames):
    """Expand sampled frame scores back to all n_frames."""
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = list(positions)
    if pos[-1] != n_frames:
        pos = pos + [n_frames]
    for i in range(len(pos) - 1):
        l, r = pos[i], pos[i + 1]
        val = scores[i] if i < len(scores) else 0.0
        frame_scores[l:r] = val
    return frame_scores



class PipelineVisualizer:
    """Drop-in visualizer for the MARs pipeline — light mode, ordered output."""

    def __init__(self, output_dir: str = "viz_output"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # 01. Feature self-similarity 

    def plot_feature_similarity(
        self,
        features,
        video_name: str = "video",
        max_frames: int = 300,
    ) -> None:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        T = features.shape[0]
        if T > max_frames:
            idx = np.linspace(0, T - 1, max_frames, dtype=int)
            features = features[idx]

        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        sim = (features / norms) @ (features / norms).T

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle(f"Feature Self-Similarity — {video_name}",
                     fontsize=13, fontweight="bold")
        im = ax.imshow(sim, cmap=_CMAP, vmin=-1, vmax=1, aspect="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        plt.colorbar(im, cax=cax, label="Cosine similarity")
        ax.set_xlabel("Frame index (subsampled)")
        ax.set_ylabel("Frame index (subsampled)")
        fig.tight_layout()
        _save(fig, self.out, "feature_similarity", video_name)

    # 02. Frame-level importance scores
    def plot_frame_scores(
        self,
        scores,
        n_frames: int,
        positions,
        shot_bound=None,
        summary=None,
        video_name: str = "video",
    ) -> None:
        scores    = np.array(scores, dtype=np.float32)
        positions = np.array(positions, dtype=np.int32)
        frame_scores = _interpolate_scores(scores, positions, n_frames)
        xs = np.arange(n_frames)

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 6),
            gridspec_kw={"height_ratios": [4, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.06)
        fig.suptitle(f"Frame Importance Scores — {video_name}",
                     fontsize=14, fontweight="bold", y=0.98)

        ax_score, ax_sel = axes
        ax_score.fill_between(xs, frame_scores, alpha=0.18, color=PALETTE["accent1"])
        ax_score.plot(xs, frame_scores, lw=1.0, color=PALETTE["accent1"], label="Frame score")
        ax_score.vlines(positions, 0, frame_scores[positions],
                        lw=0.4, alpha=0.35, color=PALETTE["subtext"])

        if shot_bound is not None:
            for shot in shot_bound:
                ax_score.axvline(shot[0], lw=0.7, ls="--",
                                 color=PALETTE["accent2"], alpha=0.7)
            ax_score.axvline(shot_bound[-1][1], lw=0.7, ls="--",
                             color=PALETTE["accent2"], alpha=0.7, label="Shot boundary")

        ax_score.set_ylabel("Importance score")
        ax_score.set_ylim(bottom=0)
        ax_score.grid(axis="y")
        ax_score.legend(loc="upper right", fontsize=8)

        if summary is not None:
            summary_arr = np.array(summary, dtype=np.float32)
            ax_sel.fill_between(xs, summary_arr, step="post",
                                color=PALETTE["accent3"], alpha=0.75, label="Selected")
            ax_sel.set_ylabel("Selected", fontsize=8)
            ax_sel.set_ylim(0, 1.4)
            ax_sel.set_yticks([])
            ax_sel.legend(loc="upper right", fontsize=8)
        else:
            ax_sel.set_visible(False)

        ax_sel.set_xlabel("Frame index")
        fig.tight_layout()
        _save(fig, self.out, "frame_scores", video_name)

    # 03. Shot importance + knapsack selection

    def plot_shot_importance(
        self,
        shot_bound,
        shot_scores,
        selected_shots,
        video_name: str = "video",
    ) -> None:
        n = len(shot_scores)
        shot_scores  = np.array(shot_scores, dtype=np.float32)
        selected_set = set(selected_shots)
        lengths = [sb[1] - sb[0] + 1 for sb in shot_bound]
        colors  = [PALETTE["accent1"] if i in selected_set else PALETTE["accent4"]
                   for i in range(n)]
        yticks  = np.arange(n)

        fig, (ax_imp, ax_len) = plt.subplots(
            1, 2, figsize=(13, max(4, n * 0.35)),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        fig.suptitle(f"Shot Importance & Selection — {video_name}",
                     fontsize=13, fontweight="bold", y=1.01)

        ax_imp.barh(yticks, shot_scores, color=colors, height=0.7, edgecolor="none")
        ax_imp.set_yticks(yticks)
        ax_imp.set_yticklabels([f"S{i}" for i in range(n)], fontsize=7)
        ax_imp.set_xlabel("Mean importance score")
        ax_imp.set_title("Importance (knapsack input)", fontsize=10)
        ax_imp.invert_yaxis()
        ax_imp.grid(axis="x")

        ax_len.barh(yticks, lengths, color=colors, height=0.7, edgecolor="none", alpha=0.7)
        ax_len.set_yticks(yticks)
        ax_len.set_yticklabels([])
        ax_len.set_xlabel("Shot length (frames)")
        ax_len.set_title("Length", fontsize=10)
        ax_len.invert_yaxis()
        ax_len.grid(axis="x")

        patches = [
            mpatches.Patch(color=PALETTE["accent1"], label="Selected"),
            mpatches.Patch(color=PALETTE["accent4"], label="Dropped"),
        ]
        fig.legend(handles=patches, loc="upper right", fontsize=9)
        fig.tight_layout()
        _save(fig, self.out, "shot_importance", video_name)

    # 04. Summary segment overlay 

    def plot_summary_selection(
        self,
        summary,
        shot_bound,
        frame_scores,
        gt_summary=None,
        video_name: str = "video",
    ) -> None:
        n     = len(summary)
        xs    = np.arange(n)
        ratio = summary.sum() / n * 100

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle(
            f"Summary Selection — {video_name}  [compression {ratio:.1f}%]",
            fontsize=13, fontweight="bold",
        )
        ax.plot(xs, frame_scores, lw=1.0, color=PALETTE["accent1"], alpha=0.8, label="Score")
        ax.fill_between(xs, frame_scores * summary, alpha=0.35,
                        color=PALETTE["accent1"], label="Predicted summary")

        if gt_summary is not None:
            gt = np.array(gt_summary, dtype=np.float32)
            ax.fill_between(xs, gt * frame_scores.max(), alpha=0.2,
                            color=PALETTE["accent3"], label="Ground truth")

        for shot in shot_bound:
            ax.axvline(shot[0], lw=0.5, ls=":", color=PALETTE["accent2"], alpha=0.5)

        ax.set_xlabel("Frame index")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(axis="y")
        fig.tight_layout()
        _save(fig, self.out, "summary_selection", video_name)

    # 05. Predicted vs Ground Truth labels

    # --- MODIFICATION 2: Added gt_scores parameter and plotted it ---
    def plot_predicted_vs_gt(
        self,
        summary,        # binary array (n_frames,) — model prediction
        gt_summary,     # binary array (n_frames,) — ground truth
        frame_scores,   # dense predicted scores (n_frames,)
        gt_scores,      # NEW: dense ground truth scores (n_frames,)
        video_name: str = "video",
    ) -> None:
        """
        Side-by-side comparison: predicted labels vs ground truth labels.
        Also shows overlap (TP), missed (FN), and false positives (FP).
        """
        summary    = np.array(summary,    dtype=np.float32)
        gt_summary = np.array(gt_summary, dtype=np.float32)
        n  = len(summary)
        xs = np.arange(n)

        tp = summary * gt_summary            # True Positive  — كلاهم اختارهم
        fp = summary * (1 - gt_summary)      # False Positive — اختاره الموديل بس
        fn = (1 - summary) * gt_summary      # False Negative — اختاره الـ GT بس

        precision = tp.sum() / (summary.sum() + 1e-8)
        recall    = tp.sum() / (gt_summary.sum() + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        fig, axes = plt.subplots(4, 1, figsize=(14, 9),
                                 sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1, 1]})
        fig.subplots_adjust(hspace=0.08)
        fig.suptitle(
            f"Predicted vs Ground Truth — {video_name}\n"
            f"Precision={precision:.3f}   Recall={recall:.3f}   F1={f1:.3f}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        ax_score, ax_pred, ax_gt, ax_diff = axes

        # --- MODIFIED SCORE CURVE SECTION ---
        # 1. Predicted Score (Solid Blue)
        ax_score.plot(xs, frame_scores, lw=1.2, color=PALETTE["accent1"], alpha=0.9, label="Predicted Score")
        ax_score.fill_between(xs, frame_scores, alpha=0.15, color=PALETTE["accent1"])
        
        # 2. Ground Truth Score (Dashed Green)
        ax_score.plot(xs, gt_scores, lw=1.5, color=PALETTE["accent3"], alpha=0.8, label="Ground Truth Score")
        ax_score.fill_between(xs, gt_scores, alpha=0.10, color=PALETTE["accent3"])
        
        ax_score.set_ylabel("Score")
        ax_score.legend(loc="upper right", fontsize=9)
        ax_score.grid(axis="y")
        # ------------------------------------

        # Predicted
        ax_pred.fill_between(xs, summary, step="post",
                             color=PALETTE["accent5"], alpha=0.75)
        ax_pred.set_ylabel("Predicted", fontsize=9)
        ax_pred.set_ylim(0, 1.5)
        ax_pred.set_yticks([])

        # Ground truth
        ax_gt.fill_between(xs, gt_summary, step="post",
                           color=PALETTE["accent3"], alpha=0.75)
        ax_gt.set_ylabel("Ground Truth", fontsize=9)
        ax_gt.set_ylim(0, 1.5)
        ax_gt.set_yticks([])

        # Diff: TP green, FP red, FN orange
        ax_diff.fill_between(xs, tp,   step="post", color=PALETTE["accent3"],  alpha=0.8, label="TP (correct)")
        ax_diff.fill_between(xs, fp,   step="post", color=PALETTE["accent4"],  alpha=0.7, label="FP (extra)")
        ax_diff.fill_between(xs, fn,   step="post", color=PALETTE["accent2"],  alpha=0.7, label="FN (missed)")
        ax_diff.set_ylabel("Overlap", fontsize=9)
        ax_diff.set_ylim(0, 1.5)
        ax_diff.set_yticks([])
        ax_diff.set_xlabel("Frame index (raw frames)")
        ax_diff.legend(loc="upper right", fontsize=8, ncol=3)

        fig.tight_layout()
        _save(fig, self.out, "predicted_vs_gt", video_name)

    # 05b. Sampled-Frames Analytical View (NEW)

    # --- MODIFICATION 3: Added entirely new analytical plot ---
    def plot_sampled_scores(
        self,
        sampled_scores,       # array (n_steps,) - predicted
        gt_sampled_scores,    # array (n_steps,) - ground truth
        video_name: str = "video",
    ) -> None:
        """
        Plots the raw, non-interpolated importance scores exactly as they were 
        generated/labeled at the sampled frames step.
        """
        n = len(sampled_scores)
        xs = np.arange(n)

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.suptitle(f"Sampled-Frames Analytical View — {video_name}",
                     fontsize=13, fontweight="bold")

        # Plot Ground Truth (Step Function)
        ax.step(xs, gt_sampled_scores, where='post', lw=1.5,
                color=PALETTE["accent4"], alpha=0.85, label="Ground Truth (Step)")

        # Plot Predicted Score (Line Plot)
        ax.plot(xs, sampled_scores, lw=1.5,
                color=PALETTE["accent1"], alpha=0.9, label="Predicted Score (Line)")

        ax.set_xlabel("Frame index (subsampled)")
        ax.set_ylabel("Importance Score")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y")

        fig.tight_layout()
        _save(fig, self.out, "sampled_scores", video_name)
    # ----------------------------------------------------------

    # 06. Keyframe contact sheet 

    def extract_keyframes(
        self,
        video_path,
        summary,
        video_name: str = "video",
        max_frames: int = 20,
        scores=None,
    ) -> None:
        video_path = str(video_path)
        summary    = np.array(summary)
        selected_indices = np.where(summary == 1)[0]

        if len(selected_indices) == 0:
            print("  [viz] No frames selected — skipping keyframe sheet.")
            return

        if len(selected_indices) > max_frames:
            idx = np.round(np.linspace(0, len(selected_indices) - 1, max_frames)).astype(int)
            selected_indices = selected_indices[idx]

        cap      = cv2.VideoCapture(video_path)
        frames   = {}
        frame_idx = 0
        sel_set  = set(selected_indices.tolist())

        with tqdm(total=summary.shape[0], ncols=80,
                  desc="extracting keyframes", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in sel_set:
                    frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_idx += 1
                pbar.update(1)
        cap.release()

        if not frames:
            print("  [viz] Could not read frames from video.")
            return

        n    = len(frames)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 3.2, rows * 2.4 + 0.6))
        fig.patch.set_facecolor(PALETTE["bg"])
        fig.suptitle(f"Keyframes — {video_name}  [{n} shown]",
                     fontsize=13, fontweight="bold")

        for i, (fidx, frame) in enumerate(sorted(frames.items())):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(frame)
            title = f"#{fidx}"
            if scores is not None and fidx < len(scores):
                title += f"  s={scores[fidx]:.2f}"
            ax.set_title(title, fontsize=7, color=PALETTE["text"])
            ax.axis("off")

        fig.tight_layout()
        _save(fig, self.out, "keyframes", video_name)

    # 07. Evaluation results 

    def plot_evaluation_results(
        self,
        kendalls,
        spears,
        dataset_name: str = "Dataset",
    ) -> None:
        n     = len(kendalls)
        xs    = np.arange(1, n + 1)
        k_arr = np.array(kendalls, dtype=np.float32)
        s_arr = np.array(spears,   dtype=np.float32)

        fig, (ax_k, ax_s) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Evaluation Results — {dataset_name}",
                     fontsize=13, fontweight="bold")

        for ax, vals, label, color in [
            (ax_k, k_arr, "Kendall τ",   PALETTE["accent1"]),
            (ax_s, s_arr, "Spearman ρ",  PALETTE["accent3"]),
        ]:
            bars = ax.bar(xs, vals, color=color, edgecolor="white",
                          linewidth=0.5, alpha=0.85, width=0.6)
            ax.axhline(vals.mean(), ls="--", lw=1.5,
                       color=PALETTE["text"], alpha=0.6,
                       label=f"Mean {vals.mean():.3f}")
            ax.fill_between(
                [0.5, n + 0.5],
                vals.mean() - vals.std(),
                vals.mean() + vals.std(),
                alpha=0.10, color=color,
            )
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color=PALETTE["text"])
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Split {i}" for i in xs], fontsize=9)
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)
            ax.grid(axis="y")

        fig.tight_layout()
        prefix   = _PREFIX["evaluation_results"]
        filename = f"{prefix}_{dataset_name}_evaluation_results.png"
        path     = str(self.out / filename)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
        plt.close(fig)
        print(f"  [viz] saved → {path}")

    # 08. Pipeline summary card 

    def plot_pipeline_summary(
        self,
        video_name: str,
        n_frames: int,
        n_shots: int,
        n_selected_shots: int,
        n_summary_frames: int,
        kendall: float = None,
        spear:   float = None,
    ) -> None:
        fig = plt.figure(figsize=(9, 5))
        fig.suptitle(f"Pipeline Summary — {video_name}",
                     fontsize=14, fontweight="bold")
        ax = fig.add_subplot(111)
        ax.axis("off")

        stats = [
            ("Total frames",      f"{n_frames:,}"),
            ("Total shots",       f"{n_shots:,}"),
            ("Selected shots",    f"{n_selected_shots:,}"),
            ("Summary frames",    f"{n_summary_frames:,}"),
            ("Compression ratio", f"{n_summary_frames/n_frames*100:.1f}%"),
        ]
        if kendall is not None:
            stats.append(("Kendall τ",  f"{kendall:.4f}"))
        if spear is not None:
            stats.append(("Spearman ρ", f"{spear:.4f}"))

        tbl = ax.table(
            cellText=[[s, v] for s, v in stats],
            colLabels=["Metric", "Value"],
            cellLoc="center", loc="center",
            bbox=[0.15, 0.0, 0.7, 1.0],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(PALETTE["accent1"])
                cell.set_text_props(color="white", fontweight="bold")
            elif c == 1:
                cell.set_facecolor("#e8f0fb")
                cell.set_text_props(color=PALETTE["accent1"])
            else:
                cell.set_facecolor(PALETTE["panel"])
                cell.set_text_props(color=PALETTE["text"])
            cell.set_edgecolor(PALETTE["border"])

        fig.tight_layout()
        _save(fig, self.out, "pipeline_summary", video_name)

    # Attention heatmap 

    def plot_attention_heatmap(
        self,
        attention_weights,
        video_name: str = "video",
        layer_name: str = "L0",
    ) -> None:
        if isinstance(attention_weights, torch.Tensor):
            attn = attention_weights.detach().cpu().numpy()
        else:
            attn = np.array(attention_weights)

        if attn.ndim == 2:
            attn = attn[np.newaxis]

        n_heads = attn.shape[0]
        cols    = min(n_heads + 1, 5)
        rows    = (n_heads + 1 + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 3.5, rows * 3.2))
        fig.suptitle(f"Attention Weights {layer_name} — {video_name}",
                     fontsize=13, fontweight="bold")

        for h in range(n_heads):
            ax = fig.add_subplot(rows, cols, h + 1)
            im = ax.imshow(attn[h], cmap=_CMAP, aspect="auto")
            ax.set_title(f"Head {h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax_mean = fig.add_subplot(rows, cols, n_heads + 1)
        im_m    = ax_mean.imshow(attn.mean(0), cmap=_CMAP, aspect="auto")
        ax_mean.set_title("Mean", fontsize=9)
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])
        plt.colorbar(im_m, ax=ax_mean, fraction=0.046, pad=0.04)

        fig.tight_layout()
        key      = f"attention_{layer_name}"
        prefix   = "00"
        filename = f"{prefix}_{video_name}_{key}.png"
        path     = str(self.out / filename)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
        plt.close(fig)
        print(f"  [viz] saved → {path}")

    def plot_paper_style_summary(
            self,
            video_path: str,
            gt_scores,          # The background shape (e.g., ground truth dense scores)
            summary,            # The binary predictions (0/1)
            shot_bound,         # KTS boundaries
            video_name: str = "video",
            bg_color: str = "#d3d3d3",    # Light grey (VASNet style)
            hl_color: str = "#00bfff",    # Deep sky blue (VASNet style highlight)
    ) -> None:
        """
        Replicates the standard paper visualization: a step-plot background of scores,
        highlighted selected segments, and representative keyframes drawn below 
        with connecting lines.
        """
        gt_scores = np.array(gt_scores, dtype=np.float32)
        summary = np.array(summary, dtype=np.int8)
        n_frames = len(gt_scores)
        xs = np.arange(n_frames)

        # 1. Identify selected KTS shots and their consistent midpoint frames
        selected_shots = []
        for shot in shot_bound:
            start, end = shot[0], shot[1]
            # If any frame in this shot is selected in the summary mask
            if summary[start:end+1].any():
                mid_frame = (start + end) // 2  # <--- The Consistency Anchor
                selected_shots.append({
                    "start": start,
                    "end": end,
                    "mid": mid_frame
                })

        num_selected = len(selected_shots)
        if num_selected == 0:
            print(f"  [viz] No segments selected for {video_name}, skipping paper plot.")
            return

        # 2. Extract the RGB frames
        cap = cv2.VideoCapture(str(video_path))
        for shot in selected_shots:
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot["mid"])
            ret, frame = cap.read()
            if ret:
                shot["image"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                shot["image"] = np.zeros((100, 100, 3), dtype=np.uint8) # fallback black square
        cap.release()

        # 3. Set up the Figure Layout
        # Top row is the plot (span all columns), Bottom row is the images
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, num_selected, height_ratios=[2.5, 1], hspace=0.3)
        
        ax_main = fig.add_subplot(gs[0, :])
        ax_images = [fig.add_subplot(gs[1, i]) for i in range(num_selected)]

        # 4. Draw the Main Plot (Scores & Highlights)
        # Background Ground Truth
        ax_main.fill_between(xs, gt_scores, step="post", color=bg_color, alpha=0.9)
        
        # Highlight Selected Segments
        highlight = gt_scores * summary
        ax_main.fill_between(xs, highlight, step="post", color=hl_color, alpha=1.0)
        
        ax_main.set_xlim(0, n_frames)
        ax_main.set_ylim(0, gt_scores.max() * 1.1)
        ax_main.axis('off') # Remove borders and ticks for clean paper look

        # 5. Draw the Images and Connecting Lines
        for i, shot in enumerate(selected_shots):
            ax_img = ax_images[i]
            ax_img.imshow(shot["image"])
            ax_img.axis('off')

            # Draw the connecting line from the plot to the image
            # Point A: The midpoint x-coordinate on the main plot, at y=0
            # Point B: The top center of the image subplot
            con = ConnectionPatch(
                xyA=(shot["mid"], 0), coordsA=ax_main.transData,
                xyB=(0.5, 1.0), coordsB=ax_img.transAxes,
                color="red", lw=1.2
            )
            fig.add_artist(con)

        fig.suptitle(f"Video Summary — {video_name}", fontsize=14, fontweight="bold")
        
        # Save
        _save(fig, self.out, "paper_style_summary", video_name)        


# Standalone CLI 

def _cli_demo(args):
    import sys
    
    sys.path.insert(0, ".")
    sys.argv = [sys.argv[0], '--dataset_name', args.dataset_name, '--data_path', args.hdf]
    
    from config import get_config
    from generate_summary import generate_summary
    from model import set_model

    config = get_config()
    viz    = PipelineVisualizer(output_dir=args.output_dir)

    with h5py.File(args.hdf, "r") as hdf:
        video_num    = args.video_num
        features_np  = np.array(hdf[f"{video_num}/features"])
        shot_bound   = np.array(hdf[f"{video_num}/change_points"])
        n_frames     = int(np.array(hdf[f"{video_num}/n_frames"]))
        positions    = np.array(hdf[f"{video_num}/picks"])
        user_summary = np.array(hdf[f"{video_num}/user_summary"])
        # --- MODIFICATION 4: Extract gtscore for the new plot ---
        gt_sampled_scores = np.array(hdf[f"{video_num}/gtscore"])

    # Raw frame GT Score (averaging users) | (its not accurate to use the user_summary for comparing it to predicted gtscores)
    # gt_dense_scores = user_summary.mean(0)

    # 2. Interpolate the sampled gtscores for the line plot (Apples-to-Apples) | (for comparing it with the predicted gtscores)
    gt_dense_scores = _interpolate_scores(gt_sampled_scores, positions, n_frames)
    
    # 3. Keep using the raw user_summary for the binary evaluation bars!
    gt_summary = (user_summary.mean(0) > 0.5).astype(np.int8)

    # 01
    viz.plot_feature_similarity(features_np, video_name=video_num)

    model = set_model(
        conformer_model_dim=config.conformer_model_dim,
        conformer_nhead=config.conformer_nhead,
        conformer_num_blocks=config.conformer_num_blocks,
        conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
        conformer_dropout=config.conformer_dropout,
        conformer_order=config.conformer_order,
    )
    weight_path = args.weights or f"./weights/{config.dataset_name}/split1.pt"
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    features_t = torch.from_numpy(features_np).float()
    inp = features_t.unsqueeze(0)
    with torch.no_grad():
        output = model(inp)
    scores = output.squeeze().cpu().numpy().tolist()

    dense_scores  = _interpolate_scores(np.array(scores), positions, n_frames)
    summary       = generate_summary([shot_bound], [scores], [n_frames], [positions])[0]
    shot_scores   = [dense_scores[s[0]:s[1]+1].mean().item() for s in shot_bound]
    selected_shots = [i for i, s in enumerate(shot_bound) if summary[s[0]:s[1]+1].any()]

    # 02
    # viz.plot_frame_scores(scores, n_frames, positions,
    #                       shot_bound=shot_bound, summary=summary,
    #                       video_name=video_num)

    # 03
    # viz.plot_shot_importance(shot_bound, shot_scores, selected_shots,
    #                          video_name=video_num)
    # 04
    # viz.plot_summary_selection(summary, shot_bound, dense_scores,
    #                            gt_summary=gt_summary, video_name=video_num)
    # 05 - The Presentation View
    # viz.plot_predicted_vs_gt(summary, gt_summary, dense_scores, gt_dense_scores,
    #                          video_name=video_num)
    # 05b - The Analytical View (NEW)
    # viz.plot_sampled_scores(np.array(scores), gt_sampled_scores, 
    #                         video_name=video_num)
    # 06
    if args.video:
        viz.extract_keyframes(args.video, summary,
                              video_name=video_num, scores=dense_scores)
        viz.plot_paper_style_summary(
            video_path=args.video,
            gt_scores=gt_dense_scores, # The gray background 
            summary=summary,           # The blue highlights
            shot_bound=shot_bound,
            video_name=video_num,
            bg_color="#d3d3d3",        # Change to blue for dppLSTM style
            hl_color="#00bfff"         # Change to red/green for dppLSTM style
        )

    # viz.plot_pipeline_summary(
    #     video_name=video_num,
    #     n_frames=n_frames,
    #     n_shots=len(shot_bound),
    #     n_selected_shots=len(selected_shots),
    #     n_summary_frames=int(summary.sum()),
    # )
    print(f"\n[viz] All plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARs Pipeline Visualizer")
    parser.add_argument("--video",      type=str, default=None)
    parser.add_argument("--hdf",        type=str, required=True)
    parser.add_argument("--video_num",  type=str, default="video_1")
    parser.add_argument("--weights",    type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="viz_output")
    parser.add_argument("--dataset_name", type=str, default="SumMe", choices=["SumMe", "TVSum"])
    args = parser.parse_args()
    _cli_demo(args)
import cv2
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm

from config import get_config
from generate_summary import generate_summary
from model import set_model
from video_helper import VideoPreprocessor
from visualization import PipelineVisualizer, _interpolate_scores

viz = PipelineVisualizer("viz_output")


def pick_frames(video_path, selections):
    cap = cv2.VideoCapture(str(video_path))
    frames   = []
    n_frames = 0

    with tqdm(total=len(selections), ncols=90,
              desc="selecting frames", unit='frame', leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if selections[n_frames]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            n_frames += 1
            pbar.update(1)

    cap.release()
    return frames


def produce_video(save_path, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    for frame in tqdm(frames, total=len(frames), ncols=90,
                      desc="generating videos", leave=False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()


def main():
    config = get_config()

    out_dir = Path(config.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_proc = VideoPreprocessor(
        sample_rate=config.sample_rate,
        device=config.device
    )

    if config.input_is_file:
        video_paths = [Path(config.file_path)]
    else:
        video_paths = sorted(Path(config.dir_path).glob(f'*.{config.ext}'))

    model = set_model(
        conformer_model_dim=config.conformer_model_dim,
        conformer_nhead=config.conformer_nhead,
        conformer_num_blocks=config.conformer_num_blocks,
        conformer_conv_kernel_sizes=config.conformer_conv_kernel_sizes,
        conformer_dropout=config.conformer_dropout,
        conformer_order=config.conformer_order
    )
    model.load_state_dict(torch.load(config.weight_path, map_location='cpu'))
    model.to(config.device)
    model.eval()

    with torch.no_grad():
        for video_path in tqdm(video_paths, total=len(video_paths),
                               ncols=80, leave=False, desc="Making videos..."):

            video_name = video_path.stem

            # video_helper بيعمل plot_feature_similarity (01) جوا run()
            n_frames, features, cps, pick = video_proc.run(video_path)

            inputs  = features.to(config.device)
            inputs  = inputs.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
            outputs = model(inputs)
            predictions = outputs.squeeze().clone().detach().cpu().numpy().tolist()

            # ── احسب كل المتغيرات الأول ──────────────────────────────────
            selections   = generate_summary([cps], [predictions], [n_frames], [pick])[0]
            dense_scores = _interpolate_scores(np.array(predictions), pick, n_frames)
            shot_scores  = [dense_scores[s[0]:s[1]+1].mean().item() for s in cps]
            selected_shots = [i for i, s in enumerate(cps)
                              if selections[s[0]:s[1]+1].any()]

            # ── 02: Frame scores ──────────────────────────────────────────
            viz.plot_frame_scores(
                predictions, n_frames, pick,
                shot_bound=cps, summary=selections,
                video_name=video_name
            )

            # ── 03: Shot importance ───────────────────────────────────────
            viz.plot_shot_importance(
                cps, shot_scores, selected_shots,
                video_name=video_name
            )

            # ── 04: Summary selection ─────────────────────────────────────
            viz.plot_summary_selection(
                selections, cps, dense_scores,
                video_name=video_name
            )

            # ── 06: Keyframes ─────────────────────────────────────────────
            viz.extract_keyframes(
                video_path, selections,
                video_name=video_name,
                scores=dense_scores
            )

            # ── 08: Pipeline summary card ─────────────────────────────────
            viz.plot_pipeline_summary(
                video_name=video_name,
                n_frames=int(n_frames),
                n_shots=len(cps),
                n_selected_shots=len(selected_shots),
                n_summary_frames=int(selections.sum()),
            )

            # ── توليد الفيديو ─────────────────────────────────────────────
            frames = pick_frames(video_path=video_path, selections=selections)
            produce_video(
                save_path=f'{config.save_path}/{video_name}.mp4',
                frames=frames,
                fps=video_proc.fps,
                frame_size=(video_proc.frame_width, video_proc.frame_height)
            )


if __name__ == '__main__':
    main()
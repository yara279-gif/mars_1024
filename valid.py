#!/usr/bin/env python3
"""
Cross-validation script for Original CSTA: Alternates between training and inference 10 times
Saves only Split and Final results to txt file, grouped by dataset
"""

import os
import subprocess
import time
import datetime
from pathlib import Path
import argparse
from visualization import PipelineVisualizer

viz = PipelineVisualizer("viz_output")


def run_command_quiet(cmd):
    """Run a command quietly and return success status and output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def extract_inference_results(inference_output):
    """Extract Split and Final results from inference output, grouped by dataset"""
    lines = inference_output.strip().split('\n')
    results = []
    final_results = []
    current_dataset = None
    dataset_splits = []

    for line in lines:
        line = line.strip()

        if '[Split' in line and 'Kendall:' in line and 'Spear:' in line:
            dataset_splits.append(line)

        elif '[FINAL -' in line and 'Kendall:' in line and 'Spear:' in line:
            if 'SumMe' in line:
                current_dataset = 'SumMe'
            elif 'TVSum' in line:
                current_dataset = 'TVSum'

            if current_dataset and dataset_splits:
                results.append(f"{current_dataset} Dataset:")
                results.extend(dataset_splits)
                results.append(line)
                results.append("")
                final_results.append((current_dataset, line))

            dataset_splits = []
            current_dataset = None

    return results, final_results          # ✅ نظيفة - مفيش viz هنا


def parse_kendall_spear(final_line):
    """استخرج Kendall و Spear من سطر FINAL"""
    import re
    k = re.search(r'Kendall:([\d.]+)', final_line)
    s = re.search(r'Spear:([\d.]+)', final_line)
    kendall = float(k.group(1)) if k else 0.0
    spear = float(s.group(1)) if s else 0.0
    return kendall, spear


def main():
    """Main validation loop"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    print("Original CSTA Cross Validation")
    print("=" * 50)

    results_dir = Path("validation_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"original_csta_results_{timestamp}.txt"

    successful_runs = 0
    all_final_results = []

    # ✅ هنا بنجمع الـ kendalls و spears عشان الـ viz في الآخر
    split_kendalls = []
    split_spears = []

    with open(results_file, 'w') as f:
        f.write(f"Original CSTA Validation Results\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        for iteration in range(1, 11):
            print(f"Iteration {iteration}/10", end=" - ")

            current_seed = 123456 + iteration

            f.write(f"Iteration {iteration}/10\n")
            f.write("-" * 20 + "\n")

            print("Training...", end=" ")
            train_cmd = f"python3 train.py --epochs 200 --model_name MARs --dataset_name {args.dataset_name} --data_path {args.data_path} --seed {current_seed}"
            train_success, _, train_err = run_command_quiet(train_cmd)

            if not train_success:
                print("FAILED (Training)")
                f.write(f"Training failed\n\n")
                continue

            print("Inference...", end=" ")
            infer_cmd = f"python3 inference.py --model_name MARs --dataset_name {args.dataset_name} --data_path {args.data_path} --seed {current_seed}"
            inference_success, inference_out, inference_err = run_command_quiet(infer_cmd)

            if not inference_success:
                print("FAILED (Inference)")
                f.write(f"Inference failed\n\n")
                continue

            results, final_results = extract_inference_results(inference_out)

            if results:
                print("SUCCESS")
                for result in results:
                    f.write(result + "\n")
                successful_runs += 1

                for dataset, final_line in final_results:
                    all_final_results.append((iteration, dataset, final_line))

                    # ✅ اجمع الـ kendall و spear من كل iteration
                    k, s = parse_kendall_spear(final_line)
                    split_kendalls.append(k)
                    split_spears.append(s)
            else:
                print("NO RESULTS")
                f.write("No results found\n")

            f.write("\n")
            f.flush()

        if all_final_results:
            print("\nGenerating FINAL results summary...")
            f.write("=" * 50 + "\n")
            f.write("FINAL RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            datasets = ['SumMe', 'TVSum']
            for dataset in datasets:
                dataset_finals = [(iter_num, final_line) for iter_num, ds, final_line in all_final_results if ds == dataset]

                if dataset_finals:
                    f.write(f"{dataset} Dataset - All Iterations:\n")
                    f.write("-" * 30 + "\n")
                    for iter_num, final_line in dataset_finals:
                        f.write(f"Iteration {iter_num}: {final_line}\n")
                    f.write("\n")

        f.write("=" * 50 + "\n")
        f.write(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful runs: {successful_runs}/10\n")

    print(f"Completed: {successful_runs}/10 successful runs")
    print(f"Results saved to: {results_file}")

    # ✅ هنا الصح - بعد ما الـ loop خلص و split_kendalls و args متاحين
    if split_kendalls:
        viz.plot_evaluation_results(
            split_kendalls, split_spears,
            dataset_name=args.dataset_name
        )


if __name__ == "__main__":
    main()
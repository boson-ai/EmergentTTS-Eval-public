import argparse
import json
import os
import matplotlib.pyplot as plt

# Mapping from filename fragment to human-readable model name
FILEMNAME_2_MODELNAME = {
    "full_3fe115352376ebae9107dd79d3781edd": "Gemini 2.5 Pro",
    "full_3fe115352376ebae9107dd79d3781edd_judger_gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "full_3fe115352376ebae9107dd79d3781edd_judger_gemini-2.5-flash-preview-04-17": "Gemini 2.5 Flash",
    "full_3fe115352376ebae9107dd79d3781edd_judger_gpt-4o-audio-preview-2024-12-17": "gpt-4o-audio",
    "full_3fe115352376ebae9107dd79d3781edd_judger_gpt-4o-mini-audio-preview-2024-12-17": "gpt-4o-mini-audio",
    "full_3fe115352376ebae9107dd79d3781edd_judger_Qwen2.5-Omni-7B": "Qwen 2.5 Omni",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Spearman correlations from ranking_summary.json files"
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help='Paths to ranking_summary.json files to read'
    )
    parser.add_argument(
        '--output-plot', '-o',
        required=True,
        help='Path where the bar plot PNG will be saved'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    model_names = []
    spearman_vals = []

    for filepath in args.json_files:
        # find which key matches this filename
        model = FILEMNAME_2_MODELNAME.get(os.path.basename(os.path.dirname(filepath)))
        if model is None:
            raise ValueError(f"No mapping found for file: {filepath}")

        # load spearman from JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        if 'spearman' not in data:
            raise KeyError(f"'spearman' not found in {filepath}")

        model_names.append(model)
        spearman_vals.append(data['spearman'])

    # Sort by Spearman correlation (descending)
    sorted_data = sorted(zip(spearman_vals, model_names), reverse=True)
    spearman_vals, model_names = zip(*sorted_data)

    print(f"Model names: {model_names}")
    print(f"Spearman values: {spearman_vals}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(model_names, spearman_vals, width=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Ranking Agreement by Model Judge')
    ax.set_xticklabels(model_names, rotation=45, ha='right')

    # annotate values on bars
    for bar, val in zip(bars, spearman_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    plt.savefig(args.output_plot, dpi=300)
    print(f"Saved plot to {args.output_plot}")

if __name__ == "__main__":
    main()
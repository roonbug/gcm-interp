import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

RM_INTERP_REPO = os.path.dirname(os.path.abspath(__file__))

# ===== Constants =====
steering_factors = list(range(10, 0, -1))
topk_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]
tasks = ["from_harmful_to_harmless", "from_hate_to_love", "from_verse_to_prose"]
task_dict = {
    "from_harmful_to_harmless": "Refusal\nInduction",
    "from_hate_to_love": "Sycophancy\nReduction",
    "from_verse_to_prose": "Verse\nStyle Transfer"
}
methods = ["acp", "atp", "atp-zero", "probes", "random"]
method_dict = {
    'acp': 'Full Vector\nPatching [[GCM]]',
    'atp': 'Attribution\nPatching [[GCM]]',
    'atp-zero': 'Attention Head\nKnockouts [[GCM]]',
    'probes': 'Inference-Time\nInterventions (ITI)',
    'random': 'Randomly Selected\nHeads'
}
models = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "SOLAR-10.7B-Instruct-v1.0"]
ablation_dict = {
    'steer': 'Difference in Means Steering',
    'pyreft': 'Representation Fine-Tuning',
    'mean': 'Means Steering'
}

save_dir = f"{RM_INTERP_REPO}/normalized-results/new-lineplots/"
os.makedirs(save_dir, exist_ok=True)

colors = {
    "acp":      "tab:blue",
    "atp":      "tab:green",
    "atp-zero": "tab:orange",
    "probes":   "tab:red",
    "random":   "tab:purple",
}

# ============================================================
#                     NEW LINE-PLOT PIPELINE
# ============================================================

for model_id in models:
    for ablation in ["steer", "pyreft", "mean"]:

        # Determine root directory
        if ablation in ["steer", "mean"]:
            root_dir = f"{RM_INTERP_REPO}/relevance-fluency/new-accuracies/{model_id}"
        else:
            root_dir = f"{RM_INTERP_REPO}/normalized-results/new-accuracies/{model_id}"

        print(f"\n=== MODEL {model_id} | ABLATION {ablation} ===")

        # Make 1×3 plots (one for each task)
        fig, axes = plt.subplots(
            nrows=1, ncols=3,
            figsize=(30, 8), constrained_layout=True
        )

        for ax, task in zip(axes, tasks):

            print(f"Processing task: {task}")

            # For each method, store best steering rate per top-k
            best_per_method = {m: [] for m in methods}

            for method in methods:
                method_dir = (
                    os.path.join(root_dir, task, method, "eval")
                    if ablation == "pyreft"
                    else os.path.join(root_dir, task, method, "eval_test/")
                )

                for topk in topk_values:
                    max_acc_for_topk = []

                    for sf in steering_factors:

                        # Filename pattern
                        if ablation in ["steer", "mean"]:
                            filename = (
                                f"{sf}_targeted_{ablation}_topk_{topk}_gen_accuracy_responses.json.accuracy.json"
                                if method != "random"
                                else f"{sf}_random_{ablation}_topk_{topk}_gen_accuracy_responses.json.accuracy.json"
                            )
                        else:
                            filename = (
                                f"{sf}_targeted_{ablation}_topk_{topk}_gen_accuracy_w_rf.json.accuracy.json"
                                if method != "random"
                                else f"{sf}_random_{ablation}_topk_{topk}_gen_accuracy_w_rf.json.accuracy.json"
                            )

                        filepath = os.path.join(method_dir, filename)

                        # Load accuracy
                        if os.path.exists(filepath):
                            with open(filepath, "r") as f:
                                data = json.load(f)
                                acc = data.get("gen", {}).get("q1", np.nan)
                                if acc is np.nan:
                                    acc = data.get("q1", np.nan)
                        else:
                            acc = np.nan

                        max_acc_for_topk.append(acc)

                    # store the max for this column
                    if len(max_acc_for_topk) > 0:
                        best_per_method[method].append(np.nanmax(max_acc_for_topk))
                    else:
                        best_per_method[method].append(np.nan)

            # -------------------------------------------------------
            #               DRAW LINES FOR THIS TASK
            # -------------------------------------------------------
            for method in methods:
                ax.plot(
                    topk_values,
                    best_per_method[method],
                    marker="o",
                    label=method_dict[method],
                    color=colors[method],
                    linewidth=3,
                    markersize=8,
                )

            # Axis formatting
            ax.set_title(task_dict[task], fontsize=22)
            ax.set_xlabel("Top-K fraction of concept-sensitive heads", fontsize=16)
            ax.set_ylabel("Best steering rate over steering factors", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 1)

            # Log x-axis? (optional)
            # ax.set_xscale("log")

        # Legend only once
        axes[-1].legend(loc="upper left", fontsize=14, bbox_to_anchor=(1.05, 1))

        # Save figure
        out_path = f"{save_dir}/{model_id}_{ablation}_maxlineplots.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_path}")

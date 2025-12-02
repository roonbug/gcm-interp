import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
import pandas as pd
import ast
from decimal import Decimal
import numpy as np
# Constants for steering factors, top-k values, tasks, and methods
steering_factors = list(range(10, 0, -1))
topk_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1]
tasks = ["from_harmful_to_harmless", "from_hate_to_love", "from_verse_to_prose"]
task_dict = {
    "from_harmful_to_harmless": "Refusal\nInduction",
    "from_hate_to_love": "Bias-Aware\nFeedback",
    "from_verse_to_prose": "Verse\nStyle Transfer"
}
methods = ["acp", "atp", "atp-zero", "probes", "random"]
method_dict = {
    'acp': 'Full Vector\nPatching [[CCM]]',
    'atp': 'Attribution\nPatching [[CCM]]',
    'atp-zero': 'Attention Head\nKnockouts [[CCM]]',
    'probes': 'Inference-Time\nInterventions (ITI)',
    'random': 'Randomly Selected\nHeads'
}

models = ["Qwen1.5-14B-Chat", "OLMo-2-1124-13B-DPO", "SOLAR-10.7B-Instruct-v1.0"][1:2]
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

for model_id in models:
    for plot_type in ["mmlu"]:
        root_dir = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/runs-new/{model_id}"

        # Define one colormap per task
        colormaps = [cm.get_cmap('viridis'), cm.get_cmap('viridis'), cm.get_cmap('viridis')]

        # Set up the subplot grid (heatmaps)
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(35, 20), constrained_layout=True)
        row_images = []  # for colorbars

        # ---- NEW: summary values storage ----
        summary_results = {task: {} for task in tasks}

        # Loop over tasks and methods
        for row_idx, task in enumerate(tasks):
            cmap = colormaps[row_idx]
            line_fig, line_ax = plt.subplots(figsize=(8, 6))
            for col_idx, method in enumerate(methods):
                heatmap_data = np.zeros((len(steering_factors), len(topk_values)))
                method_dir = os.path.join(root_dir, task, method, 'eval_test/')

                for i, sf in enumerate(steering_factors):
                    filename = (
                        f"0_extant_mmlu_{sf}_targeted_steer.json"
                        if method != "random"
                        else f"0_extant_mmlu_{sf}_random_steer.json"
                    )
                    filepath = os.path.join(method_dir, filename)
                    try:
                        data = []
                        with open(filepath, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line:  # skip empty lines
                                    obj = ast.literal_eval(line)  # safely parse Python dict string
                                    data.append(obj)
                                # Build index
                            index_by_topk = {}
                            for row in data:
                                if 'topk' not in row:
                                    continue
                                key = Decimal(str(row['topk']))
                                index_by_topk[key] = row

                            # Get baseline (topk = 0)
                            baseline_row = index_by_topk.get(Decimal("0"))
                            baseline = baseline_row.get("num-correct") if baseline_row else np.nan

                            # Fill heatmap_data normalized by baseline
                            for j, topk in enumerate(topk_values):
                                key = Decimal(str(topk))
                                row = index_by_topk.get(key)
                                accuracy = row.get("num-correct") if row else np.nan
                                if baseline and baseline != 0:
                                    accuracy = accuracy / baseline
                                heatmap_data[i, j] = accuracy
                    except FileNotFoundError:
                        heatmap_data[i, j] = np.nan

                # Collapse across steering factors
                line_mean_acc = np.nanmean(heatmap_data, axis=0)
                line_min_acc  = np.nanmin(heatmap_data, axis=0)
                line_max_acc  = np.nanmax(heatmap_data, axis=0)

                # Evenly spaced positions
                positions = np.arange(len(topk_values))

                # Plot mean line
                line_ax.plot(
                    positions, line_mean_acc,
                    label=method_dict[method],
                    marker="o", linestyle="-"
                )
                line_ax.fill_between(
                    positions, line_min_acc, line_max_acc,
                    alpha=0.2
                )
                line_ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
                line_ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
                # ---- Compute summary value per grid ----
                col_max = np.nanmax(heatmap_data, axis=0)   # shape (12,)
                mean_val = np.nanmean(col_max)
                std_val  = np.nanstd(col_max)
                min_val  = np.nanmin(col_max)
                max_val  = np.nanmax(col_max)

                summary_results[task][method] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val
                }

                # ---- Plot the heatmap ----
                ax = axes[row_idx, col_idx]
                norm = plt.Normalize(vmin=0, vmax=1)  # Normalize to [0, 1] for {plot_type}
                im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap=cmap, norm=norm)
                np.save(
                    f'{root_dir}/mmlu_{task}_{method}.npy',
                    heatmap_data
                )
                if col_idx == len(methods) - 1:
                    row_images.append((im, norm, cmap))

                # Add text annotations
                for i in range(len(steering_factors)):
                    for j in range(len(topk_values)):
                        val = heatmap_data[i, j]
                        if not np.isnan(val):
                            rgba = cmap(norm(val))
                            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                            text_color = 'black' if brightness > 0.5 else 'white'
                            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=15, color=text_color)

                # Axes and ticks
                if row_idx == 2:
                    ax.set_xticks(range(len(topk_values)))
                    ax.set_xticklabels(topk_values, rotation=90, fontsize=24)
                    # ax.set_xlabel("Top-K %", fontsize=24)
                else:
                    ax.set_xticks([])

                if col_idx == 0:
                    ax.set_yticks(range(len(steering_factors)))
                    ax.set_yticklabels(steering_factors, fontsize=24)
                    ax.set_ylabel("Steering Factor", fontsize=24)
                else:
                    ax.set_yticks([])

                # Method as column title (top row only)
                if row_idx == 0:
                    ax.set_title(method_dict[method], fontsize=24)
        
            # X-axis formatting: evenly spaced ticks with custom labels
            line_ax.set_xlabel("Top-K % of concept-sensitive attention heads", fontsize=14)
            line_ax.set_ylabel("MMLU Accuracy", fontsize=14)
            line_ax.set_xticks(positions)
            line_ax.set_xticklabels(topk_values, rotation=45, fontsize=12)
            line_ax.set_ylim(-0.05, 1.05)
            line_ax.set_title(f"{model_id} — {task_dict[task]}", fontsize=14)
            line_ax.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{root_dir}/mmlu_lineplot_{task}_{plot_type}.png", dpi=300, bbox_inches='tight')
            plt.close()

            first_ax = axes[row_idx, 0]
            pos = first_ax.get_position()

            # place label outside the grid
            fig.text(
                pos.x0 - 0.15,               # a bit to the left of first column
                (pos.y0 + pos.y1) / 2,       # centered vertically in the row
                task_dict[task], fontsize=28, weight='bold',
                va='center', ha='center', rotation=90  # vertical orientation if you like
            )

        # Add one colorbar per row
        for i, (im, norm, cmap) in enumerate(row_images):
            cbar = fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=24)


        # Add bold task labels
        # task_label_y_positions = [0.665 + 0.1, 0.395 + 0.1, 0.12 + 0.1]
        # for i, task in enumerate(tasks):
        #     fig.text(0.01, task_label_y_positions[i], task_dict[task].upper(),
        #              fontsize=15, weight='bold', va='center')

        # Title and save heatmap grid
        # plt.suptitle(
        #     f"Rate of successful steering\n(Steering Factor vs Top-K % of concept-sensitive attention heads)",
        #     fontsize=20
        # )
        plt.figtext(0.5, -0.02, "Top-K % of concept-sensitive attention heads", ha='center', fontsize=24)
        plt.figtext(1.01, 0.5, "MMLU accuracy", ha='center', va='center', rotation=90, fontsize=24)
        plt.savefig(f"{root_dir}/{plot_type}_heatmaps_per_task_colormaps.eps",
                    format='eps', transparent=True, dpi=300, bbox_inches='tight')
        plt.savefig(f"{root_dir}/{plot_type}_heatmaps_per_task_colormaps.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ---- NEW: Line plot of summary values ----
        x = np.arange(len(methods))
        special_markers = {
            "atp": "s",        # square
            "acp": "D",        # diamond
            "atp-zero": "^"    # triangle up
        }
        method_labels = {
            'acp': 'Act.\nPatching',
            'atp': 'Attrib.\nPatching',
            'atp-zero': 'Attn.\nKnockouts',
            'probes': 'ITI',
            'random': 'Random\nSelection'
        }
        colors = ["#00429D", "#EA0055", "#9EA323"]
        for taskidx, task in enumerate(tasks):
            means = [summary_results[task][m]["mean"] for m in methods]
            stds  = [summary_results[task][m]["std"]  for m in methods]

            plt.plot(x, means, marker="o", label=task_dict[task], color=colors[taskidx])
            plt.fill_between(x,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2, color=colors[taskidx]
                            )
            for i, m in enumerate(methods):
                if m in special_markers:
                    plt.scatter(
                        x[i], means[i],
                        marker=special_markers[m],
                        s=120,  # bigger size
                        edgecolor="black",
                        facecolor="none",  # hollow marker
                        linewidth=1.5,
                        zorder=5
                    )
        plt.xticks(x, [method_labels[i] for i in methods], rotation=45, ha='right', fontsize=15)
        plt.ylabel("Mean MMLU Accuracy", fontsize=15)
        plt.ylim(0, 1.2)
        plt.title(f"{model_id}", fontsize=15)
        plt.legend(ncols=len(tasks), loc='lower center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{root_dir}/mmlu_summary_lineplot.png", dpi=300, bbox_inches='tight')
        plt.close()



# plot_type = "gen"
# max_topk_allowed = 0.05
# fallback_thresholds = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# latex_records = []
# best_config_records = []

# for model_id in models:
#     root_dir = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/runs/{model_id}"
#     model_best = {"model": model_id, "best": {}}

#     for task in tasks:
#         row = {
#             "Model": model_id,
#             "Pinned Top-k": "-",  # Second column
#             "Task": task_dict[task]
#         }
#         task_best = {}

#         heatmaps = {}
#         method_max_per_topk = {}
#         method_avg_per_topk = {}
#         for method in methods:
#             heatmap_data = np.full((len(steering_factors), len(topk_values)), np.nan)
#             method_dir = os.path.join(root_dir, task, method, 'eval/')
#             for i, sf in enumerate(steering_factors):
#                 for j, topk in enumerate(topk_values):
#                     filename = (
#                         f"{sf}_targeted_steer_topk_{topk}_{plot_type}_accuracy_new.json"
#                         if method != "random"
#                         else f"{sf}_random_steer_topk_{topk}_{plot_type}_accuracy_new.json"
#                     )
#                     filepath = os.path.join(method_dir, filename)
#                     try:
#                         with open(filepath, 'r') as f:
#                             data = json.load(f)
#                             try:
#                                 accuracy = data.get(plot_type, {}).get("q1", np.nan)
#                             except:
#                                 accuracy = data.get("q1", np.nan)
#                             heatmap_data[i, j] = accuracy
#                     except FileNotFoundError:
#                         continue
#             heatmaps[method] = heatmap_data
#             method_max_per_topk[method] = [
#                 np.nanmax(heatmap_data[:, j]) for j in range(len(topk_values))
#             ]

#         # --- Find pinned top-k ---
#         pinned_topk = None
#         for threshold in fallback_thresholds:
#             for j, topk in enumerate(topk_values):
#                 if topk > max_topk_allowed:
#                     continue
#                 if any(method_max_per_topk[m][j] >= threshold for m in methods):
#                     pinned_topk = topk
#                     break
#             if pinned_topk is not None:
#                 break

#         if pinned_topk is not None:
#             row["Pinned Top-k"] = f"{pinned_topk:.2f}"
#             pinned_idx = topk_values.index(pinned_topk)

#             best_acc = -1
#             best_method = None
#             for method in methods:
#                 col = heatmaps[method][:, pinned_idx]
#                 acc = np.nanmax(col)
#                 if not np.isnan(acc) and acc > best_acc:
#                     best_acc = acc
#                     best_method = method

#             for method in methods:
#                 col = heatmaps[method][:, pinned_idx]
#                 acc = np.nanmax(col)
#                 if np.isnan(acc):
#                     row[method_dict[method]] = "-"
#                 else:
#                     sf_idx = np.nanargmax(col)
#                     sf_label = steering_factors[sf_idx]
#                     value = f"{acc:.2f} ({sf_label})"
#                     if method == best_method:
#                         value = f"\\textbf{{{value}}}"
#                     row[method_dict[method]] = value
#         else:
#             for method in methods:
#                 row[method_dict[method]] = "-"

#         # --- Find best top-k/sf combo for each method ---
#         for method in methods:
#             heatmap = heatmaps[method]
#             best_val = -1
#             best_topk = None
#             best_sf = None
#             for j, topk in enumerate(topk_values):
#                 if topk > max_topk_allowed:
#                     continue
#                 col = heatmap[:, j]
#                 if np.all(np.isnan(col)):
#                     continue
#                 sf_idx = np.nanargmax(col)
#                 val = col[sf_idx]
#                 if not np.isnan(val) and val > best_val:
#                     best_val = val
#                     best_topk = topk
#                     best_sf = steering_factors[sf_idx]
#             if best_topk is not None:
#                 task_best[method] = {"topk": best_topk, "sf": best_sf, "accuracy": round(best_val, 4)}
#         model_best["best"][task_dict[task]] = task_best
#         model_best["best"][task] = task_best

#         latex_records.append(row)
#     best_config_records.append(model_best)

# # --- LaTeX Table ---
# df_latex = pd.DataFrame(latex_records)
# column_format = 'lcl' + 'c' * len(methods)
# latex_table = df_latex.to_latex(index=False, escape=False, column_format=column_format)

# root_dir = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/runs/"
# with open(os.path.join(root_dir, f"./{plot_type}_steering_summary_table_v6_with_sf.tex"), "w") as f:
#     f.write(latex_table)

# # --- JSON Output ---
# json_output_path = os.path.join(root_dir, f"./{plot_type}_best_config_summary.json")
# with open(json_output_path, "w") as f:
#     json.dump(best_config_records, f, indent=2)

# for model in models:
#     root_dir = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/runs/{model}"
#     for method in methods:
#         for task in tasks:
#             method_dir = os.path.join(root_dir, task, method, 'eval/')
#             json_output_path = os.path.join(method_dir, f"{plot_type}_best_config_summary.json")
#             with open(json_output_path, "w") as f:
#                 json.dump(best_config_records, f, indent=2)
#             print(f"✅ JSON file saved at {json_output_path}")

# print("✅ LaTeX and JSON files saved.")

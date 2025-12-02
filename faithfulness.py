import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

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
    'acp': 'Full Vector Patching [[CCM]]',
    'atp': 'Attribution Patching [[CCM]]',
    'atp-zero': 'Attention Head Knockouts [[CCM]]',
    'probes': 'Linear Probes',
    'random': 'Randomly Selected Heads'
}
normalizing_metric_types = ['max', '1']

models = ["OLMo-2-1124-13B-DPO", "Qwen1.5-14B-Chat", "SOLAR-10.7B-Instruct-v1.0"]

# Top-k proportions (circuit sizes)
topk_proportions = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1])

def load_tensors(filepath):
    """
    Load pre_patch_logits and post_patch_logits tensors from given filepath.
    """
    data = torch.load(filepath).to(torch.float32)
    return data

def get_file_path(task, model, method, steering_factor, topk, mode='pre'):
    """
    Get the file path for the pre or post patch logits tensor.
    """
    target_type = 'random' if method == 'random' else 'targeted'
    if mode == 'pre':
        return f"./runs-new/{model}/{task}/{method}/eval/{steering_factor}_{target_type}_steer_topk_{topk}_pre_patch_logits.pt"
    elif mode == 'post':
        return f"./runs-new/{model}/{task}/{method}/eval/{steering_factor}_{target_type}_steer_topk_{topk}_patch_logits.pt"

metric_jsons = {
    metric_type: {
        model: {
            task: {
                method: {
                    steering_factor: {
                        topk: {
                            'm(C)': [],
                            'm(N)': [],
                            'm(phi)': [],
                        } for topk in topk_values
                    } for steering_factor in steering_factors
                } for method in methods
            } for task in tasks
        } for model in models
    } for metric_type in normalizing_metric_types
}

faithfulness_scores = {
    metric_type: {
        model: {
            task: {
                method: {
                    steering_factor: {
                        topk: None for topk in topk_values
                    } for steering_factor in steering_factors
                } for method in methods
            } for task in tasks
        } for model in models
    } for metric_type in normalizing_metric_types
}

for model in tqdm(models, desc="Processing models"):
    for task in tasks:
        for method in methods:
            for steering_factor in steering_factors:
                for topk in topk_values:
                    for metric_type in normalizing_metric_types:
                        pre_file_path = get_file_path(task, model, method, steering_factor, topk, mode='pre')
                        post_file_path = get_file_path(task, model, method, steering_factor, topk, mode='post')
                        
                        pre_tensor = load_tensors(pre_file_path)
                        post_tensor = load_tensors(post_file_path)
                        # print('pre and post tensor shapes ', pre_tensor[0].shape, post_tensor[1].shape)
                        metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(C)'] = torch.mean((post_tensor[1] - post_tensor[0])).item()
                        metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(phi)'] = torch.mean((pre_tensor[1] - pre_tensor[0])).item()

for model in tqdm(models, desc="Processing models (2)"):
    for task in tasks:
        for method in methods:
            for metric_type in normalizing_metric_types:
                for steering_factor in steering_factors:
                    m_C = []
                    m_N = 0
                    for topk in topk_values:
                        m_C.append(metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(C)'])
                    print('m(C) values: ', m_C)
                    if metric_type == 'max':
                        m_N = torch.max(torch.tensor(m_C))
                    elif metric_type == 'min':
                        m_N = torch.min(torch.tensor(m_C))
                    elif metric_type == 'avg':
                        m_N = torch.mean(torch.tensor(m_C))
                    elif metric_type == '1':
                        m_N = m_C[-1]#top-k = 1
                    else:
                        raise ValueError(f"Unknown metric type: {metric_type}")
                    
                    for topk in topk_values:
                        print('---')
                        metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(N)'] = m_N
                        faithfulness_scores[metric_type][model][task][method][steering_factor][topk] = (
                            (metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(C)'] - metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(phi)'])/
                            (m_N - metric_jsons[metric_type][model][task][method][steering_factor][topk]['m(phi)'])
                        )

print("Faithfulness scores computed successfully. ", faithfulness_scores)
for metric_type in normalizing_metric_types:
    # Define one colormap per task
    colormaps = [cm.get_cmap('viridis'), cm.get_cmap('plasma'), cm.get_cmap('cividis')]

    for model_id in models:
        root_dir = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/runs-new/{model_id}"
        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=len(tasks), ncols=len(methods), figsize=(24, 14))
        row_images = []  # For colorbars

        for row_idx, task in enumerate(tasks):
            cmap = colormaps[row_idx]
            for col_idx, method in enumerate(methods):
                # Prepare heatmap data for this task-method combo
                heatmap_data = np.zeros((len(steering_factors), len(topk_values)))

                for i, sf in enumerate(steering_factors):
                    for j, topk in enumerate(topk_values):
                        score = faithfulness_scores[metric_type][model_id][task][method][sf][topk]
                        heatmap_data[i, j] = score if score is not None else np.nan

                ax = axes[row_idx, col_idx]
                norm = plt.Normalize(vmin=0, vmax=1)  # Assuming scores are normalized in [0,1]
                im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap=cmap, norm=norm)

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
                            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7, color=text_color)

                # Axes and ticks
                if row_idx == len(tasks) - 1:
                    ax.set_xticks(range(len(topk_values)))
                    ax.set_xticklabels(topk_values, rotation=90)
                    ax.set_xlabel("Top-K %")
                else:
                    ax.set_xticks([])

                if col_idx == 0:
                    ax.set_yticks(range(len(steering_factors)))
                    ax.set_yticklabels(steering_factors)
                    ax.set_ylabel("Steering Factor")
                else:
                    ax.set_yticks([])

                # Method as column title (top row only)
                if row_idx == 0:
                    ax.set_title(method_dict[method])

        # Add one colorbar per row, placed to the right side with spacing
        colorbar_positions = [0.665, 0.395, 0.12]  # Adjust if necessary
        for i, (im, norm, cmap) in enumerate(row_images):
            cbar_ax = fig.add_axes([0.93, colorbar_positions[i], 0.015, 0.2])
            fig.colorbar(im, cax=cbar_ax)

        # Add bold task labels to the left of each row
        task_label_y_positions = [0.665 + 0.1, 0.395 + 0.1, 0.12 + 0.1]
        for i, task in enumerate(tasks):
            fig.text(0.01, task_label_y_positions[i], task_dict[task].upper(), fontsize=15, weight='bold', va='center')

        # Title and save
        plt.suptitle(f"{metric_type.upper()} Faithfulness Scores\n(Steering Factor vs Top-K %)", fontsize=20)
        plt.figtext(0.5, 0.01, "Top-K % of Concept-sensitive Attention Heads", ha='center', fontsize=20)
        plt.figtext(0.97, 0.5, "Faithfulness Score", ha='center', va='center', rotation=90, fontsize=10)
        
        plt.savefig(f"{root_dir}/faithfulness_{metric_type}_{model_id}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # After generating heatmaps for each model_id...

        # ---- NEW: Collect summary line plot data ----
        summary_results = {task: {} for task in tasks}

        for row_idx, task in enumerate(tasks):
            for col_idx, method in enumerate(methods):
                # reconstruct heatmap_data the same way you did above
                heatmap_data = np.zeros((len(steering_factors), len(topk_values)))
                for i, sf in enumerate(steering_factors):
                    for j, topk in enumerate(topk_values):
                        score = faithfulness_scores[metric_type][model_id][task][method][sf][topk]
                        heatmap_data[i, j] = score if score is not None else np.nan

                # 1. take column-wise max
                col_max = np.nanmax(heatmap_data, axis=0)   # shape: (len(topk_values),)

                # 2. average across columns
                mean_val = np.nanmean(col_max)
                std_val  = np.nanstd(col_max)

                summary_results[task][method] = {
                    "mean": mean_val,
                    "std": std_val
                }

        # ---- Plot summary line chart ----
        x = np.arange(len(methods))
        colors = ["#00429D", "#EA0055", "#9EA323"]
        special_markers = {"atp": "s", "acp": "D", "atp-zero": "^"}
        method_labels = {
            'acp': 'Act.\nPatching',
            'atp': 'Attrib.\nPatching',
            'atp-zero': 'Attn.\nKnockouts',
            'probes': 'ITI',
            'random': 'Random\nSelection'
        }

        all_stds = []
        for taskidx, task in enumerate(tasks):
            means = [summary_results[task][m]["mean"] for m in methods]
            stds  = [summary_results[task][m]["std"]  for m in methods]
            all_stds.extend((np.array(means)-np.array(stds)).tolist())
            all_stds.extend((np.array(means)+np.array(stds)).tolist())
            plt.plot(x, means, marker="o", label=task_dict[task], color=colors[taskidx])
            plt.fill_between(x, np.array(means)-np.array(stds), np.array(means)+np.array(stds),
                            alpha=0.2, color=colors[taskidx])
            for i, m in enumerate(methods):
                if m in special_markers:
                    plt.scatter(x[i], means[i],
                                marker=special_markers[m], s=120,
                                edgecolor="black", facecolor="none", linewidth=1.5, zorder=5)

        plt.xticks(x, [method_labels[m] for m in methods], rotation=45, ha="right", fontsize=15)
        plt.ylabel("Avg Max Faithfulness", fontsize=15)
        plt.ylim(min(all_stds) - 0.1, max(all_stds) + 0.1)
        plt.title(f"{model_id} — {metric_type.upper()}", fontsize=15)
        plt.legend(ncols=len(tasks), loc="lower center", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{root_dir}/summary_lineplot_{metric_type}_{model_id}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()


# for metric_type in normalizing_metric_types:
#     for model_id in models:
#         fig, axes = plt.subplots(nrows=len(tasks), ncols=len(methods), figsize=(24, 14), sharex=True, sharey=True)
#         root_dir = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/runs-new/{model_id}"

#         for row_idx, task in enumerate(tasks):
#             for col_idx, method in enumerate(methods):
#                 ax = axes[row_idx, col_idx]

#                 # Prepare faithfulness curve data (avg over steering factors)
#                 avg_faithfulness_per_topk = []
#                 for topk in topk_values:
#                     faithfulness_vals = []
#                     for sf in steering_factors:
#                         val = faithfulness_scores[metric_type][model_id][task][method][sf][topk]
#                         if val is not None:
#                             faithfulness_vals.append(val)
#                     if len(faithfulness_vals) > 0:
#                         avg_faithfulness = np.mean(faithfulness_vals)
#                     else:
#                         avg_faithfulness = np.nan
#                     avg_faithfulness_per_topk.append(avg_faithfulness)

#                 # Compute CPR and CMD using trapezoidal rule
#                 x = np.array(topk_values)
#                 y = np.array(avg_faithfulness_per_topk)

#                 # Remove NaNs
#                 mask = ~np.isnan(y)
#                 x = x[mask]
#                 y = y[mask]

#                 if len(x) == 0:
#                     continue  # Skip if no valid data

#                 CPR = np.trapz(y, x)
#                 CMD = np.trapz(1 - y, x)

#                 # Plot the faithfulness curve
#                 ax.plot(x, y, marker='o', color='blue', label='Faithfulness Curve')
#                 ax.fill_between(x, y, alpha=0.3, color='blue')

#                 # Plot CMD (area between curve and 1)
#                 ax.fill_between(x, y, 1, where=(y < 1), color='red', alpha=0.2, label='CMD Area')

#                 # Annotations
#                 ax.text(0.5, 0.05, f"CPR={CPR:.2f}\nCMD={CMD:.2f}", transform=ax.transAxes, 
#                         fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

#                 # Axes labels and ticks
#                 if row_idx == len(tasks) - 1:
#                     ax.set_xlabel(method_dict[method], fontsize=12)
#                 if col_idx == 0:
#                     ax.set_ylabel(task_dict[task], fontsize=12)

#                 ax.set_ylim(0, 1.05)
#                 ax.set_xlim(min(topk_values), max(topk_values))
#                 ax.grid(True, linestyle='--', alpha=0.5)

#                 # Add legend only once
#                 if row_idx == 0 and col_idx == len(methods) - 1:
#                     ax.legend(loc='lower right', fontsize=8)

#         # Global title and labels
#         plt.suptitle(f"{metric_type.upper()} CPR and CMD curves for {model_id}", fontsize=20)
#         plt.figtext(0.5, 0.01, "Top-K % of Concept-sensitive Attention Heads", ha='center', fontsize=15)
#         plt.figtext(0.97, 0.5, "Faithfulness", ha='center', va='center', rotation=90, fontsize=15)

#         plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.95])  # Leave space for title and labels
#         plt.savefig(f"{root_dir}/faithfulness_CPR_CMD_{metric_type}_{model_id}.png", dpi=300)
#         plt.show()

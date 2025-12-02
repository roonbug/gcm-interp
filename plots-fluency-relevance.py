import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Define models, tasks, topk, and methods
models = {
    "Qwen1.5-14B-Chat": {
        "harmful_harmless": {"0.03": 7, "0.05": 5},
        "verse_prose": {"0.03": 9, "0.05": 9},
        "hate_love": {"0.03": 5, "0.05": 3}
    },
    "SOLAR-10.7B-Instruct-v1.0": {
        "harmful_harmless": {"0.03": 10, "0.05": 7},
        "verse_prose": {"0.03": 9, "0.05": 7},
        "hate_love": {"0.03": 5, "0.05": 3}
    },
    "OLMo-2-1124-13B-DPO": {
        "harmful_harmless": {"0.03": 10, "0.05": 9},
        "verse_prose": {"0.03": 9, "0.05": 7},
        "hate_love": {"0.03": 7, "0.05": 7}
    }
}

methods = ["atp", "acp", "probes", "atp-zero", "random"]
topks = ["0.03", "0.05"]
metrics = ["fluency"]
reps_type = "targeted"
ablation_type = "steer"

# Task colors
task_colors = {
    "harmful_harmless": "#FFD6B2",
    "verse_prose": "#FFB0DA",
    "hate_love": "#B5E2B1"
}

# Model markers
model_markers = {
    "Qwen1.5-14B-Chat": "o",
    "SOLAR-10.7B-Instruct-v1.0": "s",
    "OLMo-2-1124-13B-DPO": "^"
}

model_marker_names = {
    "Qwen1.5-14B-Chat": "Qwen-14B",
    "SOLAR-10.7B-Instruct-v1.0": "Solar-10B",
    "OLMo-2-1124-13B-DPO": "Olmo-13B"
}
task_names = {
    "harmful_harmless": "Refusal\nInducement",
    "verse_prose": "Verse\nStyle-Transfer",
    "hate_love": "Sycophancy\nReduction"
}

# Method display names
method_names = {
    "atp": "CCM (Attribution Patching)",
    "acp": "CCM (Activation Patching)",
    "probes": "Linear Probes",
    "atp-zero": "CCM (Head Knockouts)",
    "random": "Random Head Patches"
}

# Collect metric data
plot_data = {method: [] for method in methods}

for model_id, tasks in models.items():
    for task, topk_values in tasks.items():
        for top_k in topks:
            steering_factor = topk_values[top_k]
            for method_name in methods:
                method = method_name
                metric_dir = 'probes' if method_name == 'probes' else 'numerator_1'
                reps_types = ['random'] if method_name == 'random' else ['targeted']
                if method_name == 'random':
                    method = 'atp'

                source, base = task.split('_')
                for metric in metrics:
                    file_path = f"./runs/{model_id}/from_{source}_to_{base}/None_eval/{method}-heads/{steering_factor}/no_base_logits/eval-test/global/{metric_dir}/{reps_types[0]}_{ablation_type}_{metric_dir}_heads_from_{source}_to_{base}_topk_{top_k}_{metric}_accuracy.json"
                    print(f"Loading {file_path}")
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if metric in data and data[metric]:
                                accuracy = list(data[metric].values())[0]
                                x_offset = 0 if metric == "fluency" else 2.5  # Space fluency to the right
                                x_pos = x_offset + (0 if top_k == "0.03" else 1)
                                plot_data[method_name].append({
                                    "x": x_pos,
                                    "topk": top_k,
                                    "y": accuracy,
                                    "metric": metric,
                                    "color": task_colors.get(task, "#000000"),
                                    "marker": model_markers[model_id]
                                })

# Plotting
for method, data_points in plot_data.items():
    plt.figure(figsize=(3, 5))
    ax = plt.gca()

    # For 'random', shrink plot area to leave space for legend
    if method == "random":
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*1.1, box.height])  # Shrink width


    for dp in data_points:
        plt.scatter(dp["x"], dp["y"], c=dp["color"], marker=dp["marker"],
                    edgecolor='black', alpha=0.7, s=200)

    # Set custom x-ticks
    main_xticks = [0, 1]
    main_xticklabels = ["3%", "5%"]
    plt.xticks(main_xticks, main_xticklabels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("% Ablated\nheads", fontsize=21)
    plt.ylabel("Accuracy", fontsize=21)
    # plt.title(f"{method_names[method]} – Fluency & Relevance")
    plt.ylim(0, 1.1)
    plt.xlim(-0.5, 1.5)
    plt.grid(True)

    # Add sub-labels under x-axis to mark relevance and fluency
    ax = plt.gca()
    if method == "random":
        ax.xaxis.set_label_coords(0.5, -0.2)
    else:
        ax.xaxis.set_label_coords(0.5, -0.15)
    # ax.text(0.15, -0.15, "Fluency", transform=ax.transAxes, ha='center', fontsize=18)
    # ax.text(0.8, -0.15, "Fluency", transform=ax.transAxes, ha='center', fontsize=18)

    # Add legends
    if method == "random":
        task_patches = [mpatches.Patch(color=color, label=task_names[task])
                        for task, color in task_colors.items()]
        model_lines = [mlines.Line2D([], [], color='black', marker=marker, linestyle='None',
                                    markersize=12, label=model_marker_names[model])
                    for model, marker in model_markers.items()]

        # First legend (Task)
        legend1 = ax.legend(handles=task_patches, title="Task",
                            loc='upper left', bbox_to_anchor=(1.05, 1),
                            fontsize=13, title_fontsize=18, borderaxespad=0.)
        ax.add_artist(legend1)

        # Second legend (Model)
        ax.legend(handles=model_lines, title="Model",
                loc='lower left', bbox_to_anchor=(1.05, 0),
                fontsize=13, title_fontsize=18, borderaxespad=0.)


    plt.tight_layout()
    plt.savefig(f"fluency_combined_{method}.png")

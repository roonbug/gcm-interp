import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

methods = ['steer', 'mean']
topks = [0.03]
steering_factors = [3, 5, 7, 9, 10]

# Simulate input data
def load_accuracy_data():
    data = {}
    for method in methods:
        metric = 'numerator_1'
        for topk in topks:
            for sf in steering_factors:
                try:
                    file_path  = f"./runs/{model_id}/from_{source_task}_to_{base_task}/None_eval/atp-heads/{sf}/no_base_logits/eval-test/global/{metric}/targeted_{method}_{metric}_{topk}_gen_accuracy.json"
                    with open(file_path, "r") as f:
                        acc = json.load(f)
                        q1 = acc['q1']
                except FileNotFoundError:
                    print(f"File path {file_path} not found")
                data[(method, topk, sf)] = q1
    return data

def load_control_data():
    data = {}
    for method in methods:
        metric = 'numerator_1'
        for topk in topks:
            for sf in steering_factors:
                try:
                    file_path  = f"./runs/{model_id}/from_{source_task}_to_{base_task}/None_eval/atp-heads/{sf}/no_base_logits/eval-test/global/{metric}/random_{method}_{metric}_{topk}_gen_accuracy.json"
                    with open(file_path, "r") as f:
                        acc = json.load(f)
                        q1 = acc['q1']
                except FileNotFoundError:
                    print(f"File not found for method atp, topk {topk}, sf {sf}. Using random value.")
                    # q1 = np.random.rand()
                data[(method, topk, sf)] = q1
    return data

# Data processing function
def process_accuracy_data():
    # Here you'd load the real files using model_id, task_name, method
    method_data = load_accuracy_data()
    control_data = load_control_data()
    
    data_matrix = []
    annotation_matrix = []

    for sf in steering_factors:
        row = []
        annotations = []
        for method in methods:
            for topk in topks:
                acc = method_data[(method, topk, sf)]
                control_acc = control_data[(method, topk, sf)]
                row.append(acc)
                annotations.append(f"{control_acc:.2f}")
        data_matrix.append(row)
        annotation_matrix.append(annotations)

    columns = [f"{m}\n{tk}" for m in methods for tk in topks]
    df_data = pd.DataFrame(data_matrix, index=steering_factors, columns=columns)
    df_annot = pd.DataFrame(annotation_matrix, index=steering_factors, columns=columns)

    return df_data, df_annot

# Plotting function (Option B)
def plot_heatmap_with_circles(df_data, control_data_dict):
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(df_data, annot=False, fmt='', cmap="viridis", cbar_kws={'label': 'Accuracy'})

    ax.set_xlabel("Methods and % of ablated heads")
    ax.set_ylabel("Steering Factor")

    # Add thicker lines between method groups
    for i in range(2, df_data.shape[1], 2):
        ax.axvline(i, color='white', linewidth=2)

    # Overlay circles representing |accuracy - control|
    for i, sf in enumerate(steering_factors):
        for j, col in enumerate(df_data.columns):
            method, topk = col.split("\n")
            topk = float(topk)
            acc = df_data.iloc[i, j]
            control_acc = control_data_dict[(method, topk, sf)]
            diff = acc - control_acc
            radius = diff * 500  # scale factor for visualization
            if radius > 0:
                color = 'white'
                edgecolor = 'black'
            else:
                color = 'black'
                edgecolor = 'white'
            radius = abs(radius)
            ax.scatter(j + 0.5, i + 0.5, s=radius, color=color, edgecolors=edgecolor, linewidth=0.5)

    # Create a custom legend for circle sizes
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    for size in [0.01, 0.05, 0.10]:
        plt.scatter([], [], s=size * 500, color='white', edgecolors='black', label=f'Δ = {size:.2f}')
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='|Accuracy - Control|',
               loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    plt.title("Method Accuracy with Difference Circles (Control Baseline)")
    plt.tight_layout()
    plt.savefig(f"./runs/{model_id}/from_{source_task}_to_{base_task}/None_eval/heatmap_with_circles.png")

def generate_summary_table(accuracy_data, control_data, source_task, base_task):
    task_name = f"{source_task} to {base_task}"
    rows = []
    best_row = None

    for ACC in np.arange(1, 0, -0.1):
        for sf in steering_factors:
            for topk in topks:
                acc_row = {
                    "Model": model_id,
                    "Task": task_name,
                    "topk": topk,
                    "Steering Factor": sf,
                    "ATP": "N/A"
                }

                found_any = False

                for method in methods:
                    acc = accuracy_data.get((method, topk, sf), None)
                    control = control_data.get((method, topk, sf), None)

                    if acc is not None and control is not None:
                        diff = acc - control
                        acc_row[method.upper()] = f"{acc:.2f} ({diff:+.2f})"
                        if acc >= ACC:
                            found_any = True

                if found_any:
                    best_row = acc_row
                    break
            if found_any:
                break
        if found_any:
            break

    if best_row is None:
        best_row = {
            "Model": model_id,
            "Task": task_name,
            "topk": "N/A",
            "Steering Factor": "N/A",
            "ATP": "N/A"
        }

    rows.append(best_row)
    summary_df = pd.DataFrame(rows)
    print(summary_df)

for model_id in ['Qwen1.5-14B-Chat', 'SOLAR-10.7B-Instruct-v1.0', 'OLMo-2-1124-13B-DPO']:
    for source_task, base_task in [('harmful', 'harmless'), ('verse', 'prose'), ('hate', 'love')]:
        # if source_task in ['harmful']  and model_id == 'SOLAR-10.7B-Instruct-v1.0':
        #     continue
        task_name = (source_task, base_task)
        accuracy_data = load_accuracy_data()
        control_data = load_control_data()
        df_data, df_annot = process_accuracy_data()
        plot_heatmap_with_circles(df_data, control_data)
        generate_summary_table(accuracy_data, control_data, source_task, base_task)
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Dictionaries
# -----------------------------
task_dict = {
    "from_harmful_to_harmless": "Refusal\nInduction",
    "from_hate_to_love": "Sycophancy\nReduction",
    "from_verse_to_prose": "Verse\nStyle Transfer"
}

method_dict = {
    'acp': 'Full Vector\nPatching [[GCM]]',
    'atp': 'Attribution\nPatching [[GCM]]',
    'atp-zero': 'Attention Head\nKnockouts [[GCM]]',
    'probes': 'ITI',
    'random': 'Random'
}

model_dict = {
    "Qwen1.5-14B-Chat": "Qwen-14B",
    "OLMo-2-1124-13B-DPO": "OLMo-13B",
    "SOLAR-10.7B-Instruct-v1.0": "SOLAR-10B"
}

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/new-accuracies/plots/steering_results.csv")

candidates = ["acp", "atp", "atp-zero"]
baselines  = ["probes", "random"]

pair_keys = ["model_id", "ablation", "task", "steering_factor", "topk"]
records = []

# ============================================================
#             WILCOXON TESTS (ONE-SIDED)
# ============================================================
for model in df["model_id"].unique():
    for task in df["task"].unique():
        for ablation in df["ablation"].unique():

            df_cell = df[
                (df["model_id"] == model) &
                (df["task"] == task) &
                (df["ablation"] == ablation)
            ]

            for cand in candidates:
                for base in baselines:

                    df_c = df_cell[df_cell["method"] == cand].set_index(pair_keys)["accuracy"]
                    df_b = df_cell[df_cell["method"] == base].set_index(pair_keys)["accuracy"]

                    common_idx = df_c.index.intersection(df_b.index)
                    if len(common_idx) == 0:
                        continue

                    x = df_c.loc[common_idx].values
                    y = df_b.loc[common_idx].values

                    stat, p = wilcoxon(x, y, alternative="greater")

                    records.append({
                        "model": model,
                        "task": task,
                        "ablation": ablation,
                        "candidate": cand,
                        "baseline": base,
                        "raw_p": p
                    })

# Convert to DataFrame
res = pd.DataFrame(records)

# ============================================================
#         FDR CORRECTION ACROSS ALL TESTS
# ============================================================
res["raw_p"] = res["raw_p"].fillna(1.0)
_, p_corr, _, _ = multipletests(res["raw_p"], method="fdr_bh")
res["p_corr"] = p_corr

# ============================================================
#         PIVOT → ONE ROW PER CANDIDATE
# ============================================================
res_wide = res.pivot_table(
    index=["model", "task", "ablation", "candidate"],
    columns="baseline",
    values="p_corr"
).reset_index()

# Save table
wide_path = "/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/new-accuracies/plots/wilcoxon_wide_results.csv"
res_wide.to_csv(wide_path, index=False)
print(res_wide)
print(f"\nSaved wide-format results to: {wide_path}")

# ============================================================
#              HEATMAPS FOR SIGNIFICANCE
# ============================================================

df = res_wide.copy()
df["probes_sig"] = (df["probes"] < 0.05).astype(int)
df["random_sig"] = (df["random"] < 0.05).astype(int)

baselines = ["random", "probes"]
ablations = ["steer", "mean", "pyreft"]

models = df["model"].unique()
tasks  = df["task"].unique()

def build_matrix(subdf):
    mat = np.zeros((3, 2))
    for i, cand in enumerate(candidates):
        for j, base in enumerate(baselines):
            row = subdf[(subdf["candidate"] == cand)]
            if len(row) > 0:
                mat[i, j] = int(row[f"{base}"] < 0.05)
            else:
                mat[i, j] = np.nan
    return mat

output_dir = "/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/new-accuracies/plots/"

for ab in ablations:

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(tasks),
        figsize=(18, 12),
        constrained_layout=True
    )

    if len(models) == 1 and len(tasks) == 1:
        axes = np.array([[axes]])

    for i, model in enumerate(models):
        for j, task in enumerate(tasks):

            ax = axes[i, j]

            sub = df[
                (df["model"] == model) &
                (df["task"] == task) &
                (df["ablation"] == ab)
            ]

            mat = build_matrix(sub)

            sns.heatmap(
                mat,
                ax=ax,
                cmap="viridis",
                annot=False,
                fmt=".0f",
                cbar=False,
                linewidths=0.5,
                linecolor="gray",
                vmin=0,
                vmax=1
            )

            if j == 0:
                ax.set_ylabel(model_dict[model], fontsize=12, fontweight="bold")
                ax.set_yticks(np.arange(3) + 0.5)
                ax.set_yticklabels([method_dict[c] for c in candidates], rotation=0, fontsize=11)
            else:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(task_dict[task], fontsize=12, fontweight="bold")

            if i == len(models) - 1:
                ax.set_xticks(np.arange(2) + 0.5)
                ax.set_xticklabels([method_dict[b] for b in baselines], rotation=0, fontsize=11)
            else:
                ax.set_xticklabels([])

    plt.suptitle(
        f"Paired Wilcoxon Signed Rank Test (Steered using {ab})\n"
        "1 = significant (FDR<0.05), 0 = not significant\n"
        "Rows = candidate methods, Columns = baselines",
        fontsize=16
    )

    outpath = f"{output_dir}/wilcoxon_{ab}_heatmap.png"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath}")

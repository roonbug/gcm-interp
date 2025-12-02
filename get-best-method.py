import pandas as pd
import numpy as np
import json

# --- assume df already exists; if starting from CSV, uncomment:
df = df = pd.read_csv("normalized-results/new-accuracies/plots/steering_results.csv")
# 1. Filter out all rows with topk < 0.1
df_filtered = df[df["topk"] < 0.1].copy()
df_filtered = df_filtered[df_filtered['method'].isin(['acp', 'atp', 'atp-zero'])]

group_cols = ["ablation", "model_id", "task"]
results = []

print(df_filtered.groupby(group_cols))
for (ablation, model, task), sub in df_filtered.groupby(group_cols):
    best_topk = None
    best_acc_over_topk = -np.inf
    best_rows_for_group = None

    # 2. For each topk, find rows with:
    #    max-accuracy > 0.8 if any, otherwise global max-accuracy for that topk
    for topk, sub_t in sub.groupby("topk"):
        above_threshold = sub_t[sub_t["accuracy"] > 0.8]

        if not above_threshold.empty:
            max_acc = above_threshold["accuracy"].max()
            tmp = above_threshold[above_threshold["accuracy"] == max_acc]
        else:
            max_acc = sub_t["accuracy"].max()
            tmp = sub_t[sub_t["accuracy"] == max_acc]

        # NEW: tie-break by minimum steering_factor
        min_sf = tmp["steering_factor"].min()
        candidate_rows = tmp[tmp["steering_factor"] == min_sf]

        # Compare across topk
        if (max_acc > best_acc_over_topk) or (
            np.isclose(max_acc, best_acc_over_topk) and
            (best_topk is None or topk < best_topk)
        ):
            best_acc_over_topk = max_acc
            best_topk = topk
            best_rows_for_group = candidate_rows


    # 4. Build summary for this (ablation, model, task)
    best_methods = sorted(best_rows_for_group["method"].unique().tolist())

    # best_N: use max N among tied rows if N column exists, else None
    # if "steering_factor" in best_rows_for_group.columns:

    best_N = int(best_rows_for_group["steering_factor"].max())
    # else:
    #     best_N = None

    # pairs: list of {method, N, accuracy}
    pairs_list = []
    for method, mdf in best_rows_for_group.groupby("method"):
        for _, row in mdf.iterrows():
            entry = {
                "method": method,
                "accuracy": float(row["accuracy"]),
                "steering_factor": int(row["steering_factor"])
            }
            # if "steering_factor" in mdf.columns:
            # entry["steering_factor"] = int(row["steering_factor"])
            pairs_list.append(entry)

    results.append(
        {
            "ablation": ablation,
            "model": model,
            "task": task,
            "best_methods": best_methods,
            "best_topk": best_topk,
            "pairs": json.dumps(pairs_list),  # now a list, not dict
        }
    )


best_df = pd.DataFrame(results)

# Optional: inspect result
print(best_df)
best_df.to_csv(
    "normalized-results/new-accuracies/plots/best_topk_N_per_method_per_ablation.csv",
    index=False
)

import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv("normalized-results/new-accuracies/plots/steering_results.csv")

# Methods in the column order you want
methods = ["acp", "atp", "atp-zero", "probes", "random"]

# Steering locations (ablation types)
ablations = ["steer", "pyreft", "mean"]

results = []

# 95% normal-based CI multiplier
z = norm.ppf(0.975)  # ≈ 1.96

# ---------------------------------------------------------
# BUILD 27 ROWS
# ---------------------------------------------------------
for model in df["model_id"].unique():
    for task in df["task"].unique():
        for ablation in ablations:

            df_block = df[
                (df.model_id == model) &
                (df.task == task) &
                (df.ablation == ablation)
            ]

            row = {
                "model": model,
                "task": task,
                "steering_location": ablation
            }

            # For each method, compute statistics
            for method in methods:
                method_df = df_block[df_block.method == method]["accuracy"]

                if method_df.empty:
                    m = s = se = ci_low = ci_high = np.nan
                else:
                    m = method_df.mean()
                    s = method_df.std(ddof=1)
                    n = len(method_df)
                    se = s / np.sqrt(n) if n > 0 else np.nan
                    ci_low = m - z * se if n > 1 else np.nan
                    ci_high = m + z * se if n > 1 else np.nan

                # round only for saving, keep raw values for analysis
                row[f"{method}_mean"] = round(m, 4)
                row[f"{method}_std"] = round(s, 4) if not np.isnan(s) else np.nan
                row[f"{method}_se"] = round(se, 4) if not np.isnan(se) else np.nan
                row[f"{method}_ci_low"] = round(ci_low, 4) if not np.isnan(ci_low) else np.nan
                row[f"{method}_ci_high"] = round(ci_high, 4) if not np.isnan(ci_high) else np.nan

            results.append(row)

# Store first 27 rows
results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# FINAL 28th ROW (AVERAGE OF 27 ROWS)
# ---------------------------------------------------------

summary_row = {
    "model": "AVERAGE",
    "task": "",
    "steering_location": ""
}

for method in methods:

    # mean across all 27 rows
    m = results_df[f"{method}_mean"].mean()

    # for std, se, CI — average them (meta-mean)
    s = results_df[f"{method}_std"].mean()
    se = results_df[f"{method}_se"].mean()
    ci_low = results_df[f"{method}_ci_low"].mean()
    ci_high = results_df[f"{method}_ci_high"].mean()

    summary_row[f"{method}_mean"] = round(m, 4)
    summary_row[f"{method}_std"] = round(s, 4)
    summary_row[f"{method}_se"] = round(se, 4)
    summary_row[f"{method}_ci_low"] = round(ci_low, 4)
    summary_row[f"{method}_ci_high"] = round(ci_high, 4)

# Append the summary row
results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
results_df.to_csv(
    "normalized-results/new-accuracies/plots/model_task_ablation_method_averages.csv",
    index=False
)

results_df

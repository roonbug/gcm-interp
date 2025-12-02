'''##################################################
Here is a clear, publication-ready explanation of why the Wilcoxon signed-rank test and Benjamini–Hochberg FDR correction are exactly the right statistical choices for your setting.

✅ Why the Wilcoxon Signed-Rank Test Is Appropriate

Your goal is to determine whether three intervention methods:

acp

atp

atp-zero

perform better than the baseline methods:

probes

random

across many matched evaluation points:

models

tasks

ablations

steering factors

top-k values

Each evaluation point yields a paired accuracy value for both methods.

This matches the assumptions of the Wilcoxon signed-rank test:

✔ 1. The data are paired

For every candidate and baseline method, accuracies are computed on the exact same set of items:

(model,task,ablation,steering factor,top-k)
(model,task,ablation,steering factor,top-k)

This gives a pair:

(xi,yi)
(x
i
	​

,y
i
	​

)

representing the accuracy of:

candidate method = 
xi
x
i
	​


baseline method = 
yi
y
i
	​


Since they share the same evaluation context, the results are meaningfully comparable.

Thus, a paired test (not an independent test) is required.

✔ 2. The differences are not assumed to be normally distributed

Accuracy values are bounded 
[0,1]
[0,1], often clustered near ceilings, and can be highly skewed.

A paired t-test would require normality of:

di=xi−yi
d
i
	​

=x
i
	​

−y
i
	​


But here, the distribution:

is bounded

often has many ties

can be multimodal across tasks and models

The Wilcoxon signed-rank test does not assume normality, making it the correct non-parametric test.

✔ 3. The hypothesis is one-sided

Your scientific question is:

H1:candidate method performs better than baseline
H
1
	​

:candidate method performs better than baseline

i.e.,

median(xi−yi)>0
median(x
i
	​

−y
i
	​

)>0

The Wilcoxon signed-rank test allows a one-sided alternative, unlike many non-parametric tests.

This directly tests whether candidate methods systematically produce higher accuracy than baselines.

✔ 4. The test analyzes the distribution of differences, not just means

The signed-rank test uses:

the magnitude of differences

the direction (positive vs. negative)

while being robust to outliers

This gives a stronger inference than comparing only means or medians.

✅ Why Benjamini–Hochberg FDR Is Applicable

You are performing multiple hypothesis tests:

acp > probes

acp > random

atp > probes

atp > random

atp-zero > probes

atp-zero > random

Therefore, without correction, some false positives would occur by chance.

We must control for multiple comparisons.

✔ Why not Bonferroni?

Bonferroni controls the family-wise error rate (FWER), which is extremely conservative and reduces power drastically — especially with non-parametric paired tests like Wilcoxon.

It would be inappropriate because:

you're not trying to guarantee zero false positives

instead, you want to control the rate of false discoveries

you want to maintain sensitivity to detect true effects

✔ Why BH FDR works here

Benjamini–Hochberg controls:

FDR=E[false positivestotal positives]
FDR=E[
total positives
false positives
	​

]

This is the preferred correction when:

multiple related hypotheses are tested

we want to detect real improvements without being overly conservative

the tests share samples but are not perfectly independent (BH still valid)

some true effects are expected (which is your case)

BH increases statistical power, while still controlling false discoveries.

✔ Combining Wilcoxon + BH provides exactly what the analysis requires
Wilcoxon signed-rank

Paired

Non-parametric

One-sided

Compares distributions of improvements

Benjamini–Hochberg

Controls for multiple method comparisons

Preserves power

Suited for correlated tests

Together, they provide:

A statistically valid, interpretable, and appropriately powered test for
determining whether intervention methods improve accuracy over baselines
across all evaluation conditions.
'''

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/new-accuracies/plots/steering_results.csv")

# Methods
candidates = ["acp", "atp", "atp-zero"]
baselines  = ["probes", "random"]

# Pairing keys
pair_keys = ["model_id", "ablation", "task", "steering_factor", "topk"]

records = []

# ============================================================
#                GLOBAL T-TEST (PAIRED) + FDR
# ============================================================

for cand in candidates:
    for base in baselines:

        df_c = df[df["method"] == cand].set_index(pair_keys)["accuracy"]
        df_b = df[df["method"] == base].set_index(pair_keys)["accuracy"]

        # Perfect pairing
        common = df_c.index.intersection(df_b.index)
        x = df_c.loc[common].values
        y = df_b.loc[common].values

        # Paired t-test, one-sided: cand > base
        stat, p = wilcoxon(x, y, alternative="greater")

        records.append({
            "candidate": cand,
            "baseline": base,
            "comparison": f"{cand} > {base}",
            "n_pairs": len(x),
            "raw_p": p
        })

# Convert to DF
res = pd.DataFrame(records)

# Benjamini–Hochberg FDR across 6 tests
reject, p_corr, _, _ = multipletests(res["raw_p"], method="fdr_bh")
res["p_corr"] = np.round(p_corr, 4)
res["falsified_null"] = res["p_corr"] < 0.05

print(res)

# Save CSV
res.to_csv(
    "/mnt/align4_drive/arunas/rm-interp-minimal/normalized-results/new-accuracies/plots/global_wilcoxon_results.csv",
    index=False
)

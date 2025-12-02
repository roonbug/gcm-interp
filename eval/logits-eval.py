import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, energy_distance, spearmanr, ttest_rel, wilcoxon
from tqdm import tqdm
import torch
import pandas as pd
RM_INTERP_REPO = os.environ['RM_INTERP_REPO']
def overlap_coefficient(pre, post, bins=200):
    """Approximate overlap coefficient between two distributions."""
    min_val = min(pre.min(), post.min())
    max_val = max(pre.max(), post.max())
    hist_pre, edges = np.histogram(pre, bins=bins, range=(min_val, max_val), density=True)
    hist_post, _ = np.histogram(post, bins=bins, range=(min_val, max_val), density=True)
    return np.sum(np.minimum(hist_pre, hist_post)) * (edges[1] - edges[0])

def drift_report(pre, post, n_boot=1000, rng=None):
    pre = np.asarray(pre, float)
    post = np.asarray(post, float)
    assert pre.shape == post.shape

    # Core distances
    w1 = wasserstein_distance(pre, post)
    ks = ks_2samp(pre, post).statistic
    ed = energy_distance(pre, post)

    # Paired deltas
    d = post - pre
    delta_mean = d.mean()
    delta_median = np.median(d)
    delta_iqr = np.subtract(*np.percentile(d, [75, 25]))
    frac_pos = (d > 0).mean()

    # Effect size (Cohen’s d)
    pooled_std = np.sqrt((np.var(pre) + np.var(post)) / 2)
    cohens_d = (post.mean() - pre.mean()) / pooled_std if pooled_std > 0 else np.nan

    # Overlap coefficient (0 = no overlap, 1 = identical)
    ovl = overlap_coefficient(pre, post)

    # Association
    rho, _ = spearmanr(pre, post)

    # Paired tests
    t_stat, t_p = ttest_rel(post, pre)
    try:
        w_stat, w_p = wilcoxon(post, pre, zero_method='wilcox', correction=True)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    # Bootstrap CIs
    rng = np.random.default_rng(rng)
    def boot(fn):
        idx = rng.integers(0, len(pre), size=(n_boot, len(pre)))
        return np.array([fn(pre[i], post[i]) for i in idx])

    w1_ci = np.percentile(boot(wasserstein_distance), [2.5, 97.5])
    ks_ci  = np.percentile(boot(lambda a,b: ks_2samp(a,b).statistic), [2.5, 97.5])
    ed_ci  = np.percentile(boot(energy_distance), [2.5, 97.5])

    return {
        "Wasserstein_1": (w1, tuple(w1_ci)),
        "KS_stat": (ks, tuple(ks_ci)),
        "Energy_distance": (ed, tuple(ed_ci)),
        "Delta_summary": {
            "mean": float(delta_mean), "median": float(delta_median),
            "IQR": float(delta_iqr), "frac_delta>0": float(frac_pos)
        },
        "Effect_size": {"Cohens_d": float(cohens_d), "Overlap": float(ovl)},
        "Association": {"Spearman_rho": float(rho)},
        "Paired_tests": {"t_p": float(t_p), "wilcoxon_p": float(w_p)}
    }

# -------------------
# Main loop
# -------------------
results = []
for model in tqdm(['Qwen1.5-14B-Chat', 'SOLAR-10.7B-Instruct-v1.0', 'OLMo-2-1124-13B-DPO'], desc='model'):
    for N in tqdm(range(1,11), desc='N'):
        for topk in tqdm([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1], desc='topk'):
            for methods in tqdm(['acp', 'atp', 'probes', 'random', 'atp-zero'], desc='methods'):
                for task in tqdm(['harmful_to_harmless', 'verse_to_prose', 'hate_to_love'], desc='tasks'):
                    path = f"{RM_INTERP_REPO}/normalized-results/{model}/from_{task}/{methods}/eval"
                    string = 'random' if methods == 'random' else 'targeted'

                    pre = torch.load(f'{path}/{N}_{string}_steer_topk_{topk}_pre_patch_logits.pt').to(torch.float32).cpu().numpy()
                    pre_desired, pre_undesired = pre[0,:], pre[1,:]
                    post = torch.load(f'{path}/{N}_{string}_steer_topk_{topk}_patch_logits.pt').to(torch.float32).cpu().numpy()
                    post_desired, post_undesired = post[0,:], post[1,:]

                    pre = pre_undesired - pre_desired
                    post = post_undesired - post_desired

                    report = drift_report(pre, post, n_boot=1000, rng=42)

                    results.append({
                        "model": model,
                        "task": task,
                        "method": methods,
                        "topk": topk,
                        "N": N,
                        "Wasserstein": report["Wasserstein_1"][0],
                        "KS": report["KS_stat"][0],
                        "Energy": report["Energy_distance"][0],
                        "Spearman": report["Association"]["Spearman_rho"],
                        "Delta_mean": report["Delta_summary"]["mean"],
                        "Delta_median": report["Delta_summary"]["median"],
                        "Delta_IQR": report["Delta_summary"]["IQR"],
                        "Frac_delta_pos": report["Delta_summary"]["frac_delta>0"],
                        "Cohens_d": report["Effect_size"]["Cohens_d"],
                        "Overlap": report["Effect_size"]["Overlap"],
                        "t_p": report["Paired_tests"]["t_p"],
                        "wilcoxon_p": report["Paired_tests"]["wilcoxon_p"]
                    })

df = pd.DataFrame(results)
df.to_csv('logits-eval.csv', index=False)

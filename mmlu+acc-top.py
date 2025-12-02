import os
import glob
import numpy as np

# === CONFIGURATION ===
MODEL_DIRS = [
    "/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/runs-new/Qwen1.5-14B-Chat",
    # "/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/runs-new/SOLAR-10.7B-Instruct-v1.0",
    # "/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/runs-new/OLMo-2-1124-13B-DPO",
]

TASKS = ["from_harmful_to_harmless", "from_verse_to_prose", "from_hate_to_love"]
METHODS = ["atp", "acp", "atp-zero", "random", "probes"]
TOPK = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1.0]
SF = list(range(10, 0, -1))
TOPN = 5


def load_grid_pair(model_dir, task, method):
    GRID1_GLOB = f"mmlu_{task}_{method}.npy"
    GRID2_GLOB = f"heatmap_data_{task}_{method}_gen.npy"
    pattern1 = os.path.join(model_dir, f"{GRID1_GLOB}")
    pattern2 = os.path.join(model_dir, f"{GRID2_GLOB}")
    matches1 = glob.glob(pattern1)
    matches2 = glob.glob(pattern2)

    if len(matches1) != 1 or len(matches2) != 1:
        raise FileNotFoundError(
            f"Expected one match each, got {matches1}, {matches2} for task {task}, method {method} in {model_dir}"
        )

    grid1 = np.load(matches1[0])
    grid2 = np.load(matches2[0])
    return grid1, grid2


def find_top_cells(grid1, grid2, topn=5):
    combined = (grid1 + grid2) / 2.0
    flat = combined.flatten()
    top_indices = np.argsort(flat)[-topn:][::-1]
    results = []
    for idx in top_indices:
        s_idx, k_idx = np.unravel_index(idx, combined.shape)
        results.append((s_idx, k_idx, combined[s_idx, k_idx]))
    return results


def analyze_model_dir(model_dir):
    print(f"\n=== MODEL: {model_dir} ===")
    for task in TASKS:
        all_candidates = []  # collect across methods
        for method in METHODS:
            try:
                grid1, grid2 = load_grid_pair(model_dir, task, method)
                top_cells = find_top_cells(grid1, grid2, TOPN)
                # include method info for clarity
                for s_idx, k_idx, val in top_cells:
                    all_candidates.append(
                        (method, SF[s_idx], TOPK[k_idx], val)
                    )
            except FileNotFoundError as e:
                print(e)

        # take global top 5 across methods
        if all_candidates:
            all_candidates.sort(key=lambda x: x[-1], reverse=True)
            print(f"\nTask: {task} (Top {TOPN} overall across methods)")
            for method, sf, topk, val in all_candidates[:TOPN]:
                print(f"  Method {method}, Steering {sf}, Top-k {topk}, Mean {val:.4f}")


if __name__ == "__main__":
    for model_dir in MODEL_DIRS:
        analyze_model_dir(model_dir)

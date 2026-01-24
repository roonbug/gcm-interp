import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import torch
from math import ceil

def plot_violin_comparison(patch_logits, topk, reps_type, config, logit_metric, ablation_type):
    des_post = patch_logits[0]
    undes_post = patch_logits[1]
    data = [des_post.to(torch.float32).numpy(), undes_post.to(torch.float32).numpy()]
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, palette='muted')
    plt.xticks([0, 1], ['Desired', 'Undesired'], rotation=0)
    plt.title(f'Logit distribution (TopK: {int(topk*100)}%) - {reps_type}')
    plt.tight_layout()

    dir_path = f"{config.get_output_prefix()}/eval/"
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(f"{dir_path}/{config.args.N}_{reps_type}_{ablation_type}_{topk}_violin.png")
    plt.close()
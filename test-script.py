import torch
import pandas as pd
import sys
sys.path.insert(0, '/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow')
from dataclasses import dataclass, field
from typing import Optional
import os
import json
from model_handler import ModelHandler
from data_handler import DataHandler
from batch_handler import BatchHandler
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
import random

run_dict = {
    # 'Qwen/Qwen1.5-14B-Chat': {
    #     'N': 8,
    #     'topk_vals': [0.04, 0.08, 0.09]
    # },
    'allenai/OLMo-2-1124-13B-DPO': {
        'N': 10,
        'topk_vals': [0.06, 0.09, 0.1]
    },
    'upstage/SOLAR-10.7B-Instruct-v1.0': {
        'N': 8,
        'topk_vals': [0.05, 0.09, 0.1]
    }
}

for model_id in run_dict.keys():
    @dataclass
    class Args:
        prod: bool = False
        device: str = 'cuda:0'
        model_id: str = model_id
        batch_size: int = 16
        judge_id: str = 'meta-llama/Llama-3.1-70B-Instruct'
        heads: bool = True
        batch_start: int = 0
        N: int = run_dict[model_id]['N']
        patch_model: bool = False
        eval_model: bool = True
        # op_suffix: Optional[str] = 'verse_prose'
        patch_algo: Optional[str] = 'acp'
        source: Optional[str] = 'verse'
        base: Optional[str] = 'prose'
        ablation_type: Optional[str] = 'steer'
        eval_type: Optional[str] = 'None'
        eval_exps: Optional[str] = None
        eval_coh: bool = False
        random_ablate: bool = False
        consistent_ablate: bool = False
        eval_test: bool = False
        ndif_token: Optional[str] = None
        hf_token: Optional[str] = None
        pyreft: Optional[bool] = False
        vllm: Optional[bool] = False
        aidevi: Optional[bool] = False
        eval_transfer: Optional[bool] = False
        max_new_tokens: int = 256
        judge_answer_match: Optional[bool] = False
        eval_train: bool = False
        eval_test: bool = True
        eval_extant: bool = False
        eval_mean: bool = False
        data_path: Optional[str] = field(init=False)  # Set by logic, not CLI
        seed: Optional[int] = 72

        def __post_init__(self):
            model = self.model_id.split("/")[-1]
            if self.source in ('harmful', 'harmless'):
                self.data_path = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/data/{self.model_id.split('/')[-1]}/logits/harmful"
            elif self.source in ('hate', 'love'):
                self.data_path = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/data/{self.model_id.split('/')[-1]}/logits/hate"
            elif self.source in ('verse', 'prose'):
                self.data_path = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/data/{self.model_id.split('/')[-1]}/logits/verse"
            elif self.source in ('truth', 'lie'):
                self.data_path = f"/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/data/{self.model_id.split('/')[-1]}/logits/truth"
            if self.patch_algo == None:
                self.patch_algo = 'atp'

    class Config:
        def __init__(self):
            self.args = Args()
            self.setup_environment(seed=self.args.seed)

        def get_output_prefix(self):
            return self.output_prefix
        
        def setup_environment(self, seed=42):
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            os.makedirs(f'{self.set_output_prefix()}', exist_ok=True)
            # self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
            print(f'Saved config to file {self.get_output_prefix()}/config.yml')

        def set_output_prefix(self):
            model = self.args.model_id.split('/')[-1]
            # timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            if self.args.patch_model or self.args.eval_model:
                # if self.args.eval_test_dataset:
                #     with open(f"./runs/{model}/from_{self.args.source}_to_{self.args.base}/atp/eval/gen_best_config_summary.json", 'r') as f:
                #         decoded_responses = json.load(f)
                #         print(decoded_responses)
                #     try:
                #         decoded_responses = [d for d in decoded_responses if d["model"] == self.args.model_id.split('/')[-1]][0]
                #     except Exception as e:
                #         print(decoded_responses)
                #         raise ValueError(f"No responses found for model {self.args.model_id.split('/')[-1]} in the provided JSON file.")

                #     patch_algo = max(
                #         decoded_responses["best"][f"from_{self.args.source}_to_{self.args.base}"].items(),
                #         key=lambda item: round(item[1]["accuracy"], 2)
                #     )
                #     self.args.patch_algo, patch_algo_params = patch_algo
                #     print(f"Best Method: {patch_algo}, topk: {patch_algo_params['topk']}, sf: {patch_algo_params['sf']}, accuracy: {patch_algo_params['accuracy']}")

                #     topk = patch_algo_params["topk"]
                #     sf = patch_algo_params["sf"]
                #     self.args.N = 10
                self.output_prefix = f"./runs-new/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/"
            print(self.output_prefix)
            return self.output_prefix
        
    config = Config()
    print(config.args.data_path)

    model_handler = ModelHandler(config)

    data_handler = DataHandler(config, model_handler)
    batch_handler = BatchHandler(config, data_handler, 0, 16)

    import sys
    sys.path.insert(0, '/mnt/align4_drive/arunas/rm-interp/sycophancy/gen-flow/eval/')
    from eval_runner import *
    from patching_utils import PatchingUtils
    from patching import Patching
    patching = Patching(model_handler, batch_handler, config)
    patching_utils = PatchingUtils(patching)

    run_eval(config, data_handler, model_handler, batch_handler, patching_utils, which_patch='heads', topk_vals=run_dict[model_id]['topk_vals'], N=run_dict[model_id]['N'])
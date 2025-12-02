import copy
from data_handler import DataHandler
from eval.activations import mean_ablations_cache, steering_reps_cache
import torch
def get_data_path(config):
        if config.args.source in ('harmful', 'harmless'):
            config.args.data_path = f"./data/{config.args.model_id.split('/')[-1]}/logits/harmful"
        elif config.args.source in ('hate', 'love'):
            config.args.data_path = f"./data/{config.args.model_id.split('/')[-1]}/logits/hate"
        elif config.args.source in ('verse', 'prose'):
            config.args.data_path = f"./data/{config.args.model_id.split('/')[-1]}/logits/verse"
        elif config.args.source in ('truth', 'lie'):
            config.args.data_path = f"./data/{config.args.model_id.split('/')[-1]}/logits/truth"

def load_patching_reps(data_handler, model_handler, mean=True):
    model = model_handler.model
    patching_reps = {}
    for ablation in [data_handler.config.args.ablation]:
        patching_reps[ablation] = {}
        for key in ['desired', 'undesired']:
            print(f"Loading patching reps for {ablation} - {key}")
            patching_reps[ablation][key] = get_patch_activations(model, data_handler, ablation, key=key, mean=mean)
    print('Returning patching reps')
    return patching_reps

def get_patch_activations(model, data_handler, ablation_type, key='desired', mean=True):
    if ablation_type == 'mean':
        return mean_ablations_cache(model, data_handler, key=key)
    elif ablation_type == 'steer':
        return steering_reps_cache(model, data_handler, key=key, mean=mean)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

def save_patching_reps(config, patching_reps, source, base, ablation, key):
    path = f'./normalized-results/{config.args.model_id.split("/")[-1]}/steering_vectors/from_{source}_to_{base}/{ablation}_{key}.pt'
    torch.save(patching_reps, path)

def load_and_save_steering_vectors(config, model_handler):
    new_config = copy.deepcopy(config)
    for source, base in zip([('harmful', 'harmless'), ('hate', 'love'), ('verse', 'prose')]):
        new_config.args.source = source
        new_config.args.base = base
        new_config.data_path = f"./data/{new_config.args.model_id.split('/')[-1]}/logits/harmful"
        data_handler = DataHandler(new_config, model_handler)
        patching_reps = {}
        for ablation in ['mean', 'steer']:
            patching_reps[ablation] = {}
            print(f"Loading steering vectors for {source} vs {base} - {ablation}")
            for key in ['desired', 'undesired']:
                patching_reps = get_patch_activations(model_handler.model, data_handler, ablation, key=key, mean=True)
                save_patching_reps(patching_reps, source, base, ablation, key)

def normalize_steering_vectors(config, model_handler):
    model = model_handler.model
    DIM = model_handler.dim
    BEHAVIOURS = [('harmful', 'harmless'), ('hate', 'love'), ('verse', 'prose')]
    ABLATIONS = ['mean', 'steer']
    KEYS = ['desired', 'undesired']

    num_layers = len(model.model.layers)
    num_heads = model.model.config.num_attention_heads

    eps = 1e-12

    for source, base in BEHAVIOURS:
        for ablation in ABLATIONS:
            for key in KEYS:

                path = f'./normalized-results/{config.args.model_id.split("/")[-1]}/steering_vectors/from_{source}_to_{base}/{ablation}_{key}.pt'
                steering_vectors = torch.load(path).to(model.device)
                # steering_vectors shape: [num_layers, num_tokens, hidden_dim]

                normalized = []

                for layer in range(num_layers):
                    layer_vecs = steering_vectors[layer]  # [num_tokens, hidden_dim]
                    num_tokens, hidden_dim = layer_vecs.shape

                    # We'll fill this with the normalized result
                    normalized_layer = torch.empty_like(layer_vecs)

                    # Normalize each head separately
                    for head in range(num_heads):
                        head_slice = slice(DIM * head, DIM * (head + 1))

                        head_vecs = layer_vecs[:, head_slice]           # [num_tokens, head_dim]
                        norms = head_vecs.norm(dim=1, keepdim=True)     # [num_tokens, 1]

                        normalized_layer[:, head_slice] = head_vecs / (norms + eps)

                    normalized.append(normalized_layer)

                normalized = torch.stack(normalized, dim=0)  # shape [num_layers, num_tokens, hidden_dim]

                save_path = f'./normalized-results/{config.args.model_id.split("/")[-1]}/steering_vectors/from_{source}_to_{base}/{ablation}_{key}_normalized.pt'
                torch.save(normalized.cpu(), save_path)
                print(f"Saved normalized vectors to {save_path}")
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

def compute_judge_accuracy(judge_model, tokenizer, prompts, judge_items, base, source):
    print('Starting judge accuracy computation...')
    tokenizer.pad_token = tokenizer.eos_token
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    accuracy = {qid: 0 for qid in questions}
    responses = []

    conversations = []
    assert len(prompts) > 0, "No prompts provided for evaluation."
    for p in prompts:
        q = p['query']
        r = p[f'edit_{base}']
        if source in ['hate', 'harmful']:
            comparison = f"{q}\nResponse (1): {r}\nResponse (2): {p[f'old_{base}']}"
        else:
            comparison = f"{q}\nResponse: {r}"
        conversations.append(comparison)
    assert len(conversations) > 0, "No conversations to evaluate."
    batch_size = 32  # adjust for your GPU memory

    batched_inputs = []
    metadata = []  # keep track of (idx, conversation)

    # Step 1: collect all prompts + metadata
    for idx, item in enumerate(judge_items):
        for c in conversations:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': item['instruction'].replace("{conversation}", c)}],
                add_generation_prompt=True,
                tokenize=False
            )
            batched_inputs.append(prompt)
            metadata.append((idx, c))

    print(f"Judging {len(batched_inputs)} conversations total")

    # Step 2: process in batches
    num_batches = ceil(len(batched_inputs) / batch_size)

    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        input_chunk = batched_inputs[start:end]
        meta_chunk = metadata[start:end]

        # Tokenize
        tokens = tokenizer(
            input_chunk,
            return_tensors='pt',
            padding=True,
            return_attention_mask=True
        ).to(judge_model.device)

        # Generate
        outputs = judge_model.generate(
            **tokens,
            max_new_tokens=3,
            use_cache=False,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None
        )

        # Decode + evaluate
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (idx, _), d in zip(meta_chunk, decoded):
            match = re.search(r'\d+', d[-10:])
            if match and match.group() in expected_answers[idx]:
                accuracy[questions[idx]] += 1
            responses.append(d)


    for k in accuracy:
        accuracy[k] /= len(prompts)
    return accuracy, responses

def compute_judge_accuracy_test(judge_model, tokenizer, prompts, judge_items, base, source):
    print('Starting judge accuracy computation... for test wth MCQA')
    tokenizer.pad_token = tokenizer.eos_token
    questions = [j['qid'] for j in judge_items]
    expected_answers = [j['exp_answer'] for j in judge_items]
    accuracy = { evaluating: {qid: 0} for evaluating in ['old', 'edit'] for qid in questions}
    responses = {evaluating: [] for evaluating in ['old', 'edit'] }
    edit_comparisons = []
    old_comparisons = []
    correct_answers = []
    assert len(prompts) > 0, "No prompts provided for evaluation."
    for p in prompts:
        for response in ['old', 'edit']:
            r = p[f'{response}_{base}']
            correct_answer = p['answer']
            comparison = f"User: {p['query'].strip()}{r}"
            if response == 'old':
                old_comparisons.append(comparison)
            elif response == 'edit':
                edit_comparisons.append(comparison)
            correct_answers.append(correct_answer)

    for idx, item in enumerate(judge_items):
        for evaluating in ['old', 'edit']:
            if evaluating == 'old':
                conversations = old_comparisons
            elif evaluating == 'edit':
                conversations = edit_comparisons
            # print(item['instruction'])
            input_texts = [
                tokenizer.apply_chat_template([{'role': 'user', 'content': item['instruction'].replace("{conversation}", c).replace("{correct_answer}", a)}],
                add_generation_prompt=True,
                tokenize=False)
                for c, a in zip(conversations, correct_answers)
            ]
            # print('Judging ', len(input_texts), ' conversations')
            # print('Sample input text: ', input_texts[0], conversations[0], correct_answers)
            tokens = tokenizer(input_texts, return_tensors='pt', padding=True, return_attention_mask=True).to(judge_model.device)
            outputs = judge_model.generate(
                **tokens,
                max_new_tokens=3,
                use_cache=False,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j, d in enumerate(decoded):
                match = re.search(r'\d+', d[-10:])
                if match and match.group() in expected_answers[idx]:
                    accuracy[evaluating][questions[idx]] += 1
                responses[evaluating].append(d)
                    
    for evaluating in accuracy:
        for k in accuracy[evaluating]:
            accuracy[evaluating][k] /= len(prompts)
    return accuracy, responses

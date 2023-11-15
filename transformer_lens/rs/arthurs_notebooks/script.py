param_sizes = {
    "gpt2": "85M", # Extremely sus numbers lol
    "gpt2-medium": "302M",
    "gpt2-large": "708M",
    "gpt2-xl": "1.5B",
    "distilgpt2": "42M",
    "distillgpt2": "42M",
    "opt-125m": "85M",
    "opt-1.3b": "1.2B",
    "opt-2.7b": "2.5B",
    "opt-6.7b": "6.4B",
    "opt-13b": "13B",
    "gpt-neo-125m": "85M",
    "gpt-neo-1.3b": "1.2B",
    "gpt-neo-2.7b": "2.5B",
    "gpt-neo-1.3B": "1.2B",
    "gpt-neo-2.7B": "2.5B",
    "gpt-j-6B": "5.6B",
    "stanford-gpt2-small-a": "85M",
    "stanford-gpt2-small-b": "85M",
    "stanford-gpt2-small-c": "85M",
    "stanford-gpt2-small-d": "85M",
    "stanford-gpt2-small-e": "85M",
    "stanford-gpt2-medium-a": "302M",
    "stanford-gpt2-medium-b": "302M",
    "stanford-gpt2-medium-c": "302M",
    "stanford-gpt2-medium-d": "302M",
    "stanford-gpt2-medium-e": "302M",
    "pythia-70m": "19M",
    "pythia-160m": "85M",
    "pythia-410m": "302M",
    "pythia-1b": "5M",
    "pythia-1.4b": "1.2B",
    "pythia-2.8b": "2.5B",
    "pythia-6.9b": "6.4B",
    "pythia-12b": "11B",
    "pythia-70m-deduped": "19M",
    "pythia-160m-deduped": "85M",
    "pythia-410m-deduped": "302M",
    "pythia-1b-deduped": "805M",
    "pythia-1.4b-deduped": "1.2B",
    "pythia-2.8b-deduped": "2.5B",
    "pythia-6.9b-deduped": "6.4B",
    "pythia-12b-deduped": "11B",
    "solu-4l": "13M",
    "solu-6l": "42M",
    "solu-8l": "101M",
    "solu-10l": "197M",
    "solu-12l": "340M",
    "solu-4l-pile": "13M",
    "solu-6l-pile": "42M",
    "solu-8l-pile": "101M",
    "solu-10l-pile": "197M",
    "solu-12l-pile": "340M",
    "solu-1l": "3.1M",
    "solu-2l": "6.3M",
    "solu-3l": "9.4M",
    "solu-4l": "13M",
    "solu-6l": "42M",
    "solu-8l": "101M",
    "solu-10l": "197M",
    "solu-12l": "340M",
    "gelu-1l": "3.1M",
    "gelu-2l": "6.3M",
    "gelu-3l": "9.4M",
    "gelu-4l": "13M",
    "attn-only-1l": "1.0M",
    "attn-only-2l": "2.1M",
    "attn-only-3l": "3.1M",
    "attn-only-4l": "4.2M",
    "attn-only-2l-demo": "2.1M",
    "solu-1l-wiki": "3.1M",
    "solu-4l-wiki": "13M",
    # "mistralai/Mistral-7B-v0.1": "7B", # Broken, sad
    "Llama-2-7b-hf": "7B",
    "Llama-2-13b-hf": "13B",
    "llama-7b-hf": "7B",
    "llama-13b-hf": "13B",
}

models = list(param_sizes.keys())

import torch
import subprocess
import os
from itertools import product
import numpy as np
import multiprocessing

used = set()

def run_script(threshold, gpu_id, keywords):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = ["python", "/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/arthurs_notebooks/sweep_induction_ioi.py"]
    for key, value in keywords.items():
        args.append(f"--{key}={value}")
    print(" ".join(args))
    subprocess.run(args, env=env)

if __name__ == '__main__':

    num_gpus = 1 # Number of GPUs available
    num_jobs_per_gpu = 1  # Number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []
    keyword_list = []

    for model in models:
        keyword_list.append({"model": model})

    for threshold_idx, keywords in enumerate(keyword_list):
        gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
        jobs.append(pool.apply_async(run_script, (keywords, gpu_id, keywords)))
    
    for job in jobs:
        print(job, "\n")
        job.get()
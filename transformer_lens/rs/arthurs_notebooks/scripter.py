import subprocess
fpath = "/root/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/arthurs_notebooks/cspa_failures.py"

for start_index in range(20, 5000, 20):
    subprocess.run(["python", fpath, "--start-index", str(start_index), "--length", "20"])
    print("Done with", start_index)

import os
import subprocess
import numpy as np 
import multiprocessing
from math import gcd

used = set()

def run_script(threshold, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", fpath, "--start-index", str(threshold), "--length", "20"], env=env)

if __name__ == '__main__':

    num_gpus = 6 # specify the number of GPUs available
    num_jobs_per_gpu = 1 # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = list(range(2000, 500, -20))

        if not isinstance(curspace, list):
            curspace = curspace[1:-1]

        for threshold_idx, threshold in list(enumerate(curspace)):
            if threshold in used:
                continue
            used.add(threshold)

            gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
            jobs.append(pool.apply_async(run_script, (threshold, gpu_id)))

        if isinstance(curspace, list):
            break

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()
    pool.join()
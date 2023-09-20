import os
import subprocess
import multiprocessing

script_path = "/root/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/arthurs_notebooks/cspa_failures.py"
used = set()

def run_script(threshold, gpu_id, artifact_base):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", script_path, "--start-index", str(threshold), "--length", "20", "--artifact-base", artifact_base], env=env)

if __name__ == '__main__':

    num_gpus = 4  # specify the number of GPUs available
    num_jobs_per_gpu = 1  # specify the number of jobs per GPU

    pool = multiprocessing.Pool(num_gpus * num_jobs_per_gpu)
    jobs = []

    for it in range(3, int(1e6)):
        curspace = list(range(0, 2200, 20))

        if not isinstance(curspace, list):
            curspace = curspace[1:-1]

        for threshold_idx, threshold in list(enumerate(curspace)):
            if threshold in used:
                continue
            used.add(threshold)

            gpu_id = (threshold_idx // num_jobs_per_gpu) % num_gpus
            jobs.append(pool.apply_async(run_script, (threshold, gpu_id, "cspa_no_bosses")))

        if isinstance(curspace, list):
            break

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()
    pool.join()

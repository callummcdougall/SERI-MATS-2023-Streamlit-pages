import subprocess
fpath = "/root/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/arthurs_notebooks/cspa_failures.py"

for start_index in range(0, 5000, 30):
    subprocess.run(["python", fpath, "--start-index", str(start_index), "--length", "20"])
    print("Done with", start_index)
import streamlit as st

import os, sys
from pathlib import Path


DEBUG = True

import sys, os
for st_page_dir in [
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri-mats-2023-streamlit-pages")[0] + "seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("/app/seri-mats-2023-streamlit-pages")[0] + "/app/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    "/mount/src/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    "callummcdougall/seri-mats-2023-streamlit-pages/main/transformer_lens/rs/callum2/st_page",
]:
    if os.path.exists(st_page_dir):
        break
else:
    raise Exception("Couldn't find root dir")

root_dir = st_page_dir.replace("/transformer_lens/rs/callum2/st_page", "")

# We change to st_page_dir, so that we can read media (although maybe that's not necessary cause we have `ST_HTML_PATH` which we use directly)
os.chdir(st_page_dir)
ST_HTML_PATH = Path(st_page_dir) / "media"

# We make sure that the version of transformer_lens we can import from is 0th in the path
if sys.path[0] != root_dir: sys.path.insert(0, root_dir)

if DEBUG:
    print("st_page_dir:", st_page_dir)
    print("root_dir:", root_dir)
    print("sys.path:", sys.path)
    print("ST_HTML_PATH:", ST_HTML_PATH)


import platform
is_local = (platform.processor() != "")
if is_local:
    NEGATIVE_HEADS = [(10, 7), (11, 10), (10, 1)]
    HTML_PLOTS_FILENAME = "GZIP_HTML_PLOTS_b48_s61.pkl"
else:
    NEGATIVE_HEADS = [(10, 7), (11, 10)]
    HTML_PLOTS_FILENAME = "GZIP_HTML_PLOTS_b41_s51.pkl"

NEGATIVE_HEADS = sorted(NEGATIVE_HEADS)

st.markdown(
r"""
# Explore Prompts

This page was designed to help explore different prompts for GPT-2 Small, as part of Callum McDougall, Arthur Conmy & Cody Rushing's work on self-repair in LLMs. We focus on negative behaviour (specifically copy-suppression in heads 10.7 and 11.10 for GPT2-small) and backup behaviour (specifically in the IOI task).

The goals of this page are:

* Help us keep track of our work in an accessible, readable way (rather than having everything dumped into messy notebooks and directories which we'll never return to),
* Provide a sandbox environment to help us spot interesting things about the behaviour of negative heads which we might otherwise have missed (e.g. their behaviour on bigrams),
* Make our work more accesible to others.
""")
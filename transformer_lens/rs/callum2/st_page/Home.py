import streamlit as st

import os, sys
from pathlib import Path

from transformer_lens.rs.callum2.utils import ST_HTML_PATH

DEBUG = True

if DEBUG:
    st.write(os.getcwd())
    st.write(list(Path.cwd().iterdir()))
    st.write(ST_HTML_PATH)
    st.write("What's going on?")

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
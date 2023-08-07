import streamlit as st
import os, sys
from pathlib import Path

# Stuff to make the page work on my local machine
p = Path(r"C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\seri_mats_23_streamlit_pages")
if os.path.exists(str_p := str(p.resolve())):
    os.chdir(str_p)
    if (sys.path[0] != str_p):
        sys.path.insert(0, str_p)

st.markdown(
r"""
# Explore Prompts

This page was designed to help explore different prompts for GPT-2 Small, as part of Callum McDougall, Arthur Conmy & Cody Rushing's work on self-repair in LLMs. We focus on negative behaviour (specifically copy-suppression in heads 10.7 and 11.10 for GPT2-small) and backup behaviour (specifically in the IOI task).

The goals of this page are:

* Help us keep track of our work in an accessible, readable way (rather than having everything dumped into messy notebooks and directories which we'll never return to),
* Provide a sandbox environment to help us spot interesting things about the behaviour of negative heads which we might otherwise have missed (e.g. their behaviour on bigrams),
* Make our work more accesible to others.
""")
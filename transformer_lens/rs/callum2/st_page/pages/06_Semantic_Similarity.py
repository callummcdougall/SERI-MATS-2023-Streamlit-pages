import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
st.set_page_config(layout="wide")

from pathlib import Path
import plotly.express as px

from transformer_lens.rs.callum2.st_page.streamlit_styling import styling
from transformer_lens.rs.callum2.generate_st_html.utils import ST_HTML_PATH
styling()

import pandas as pd
import numpy as np
import pickle
from typing import Literal

import torch as t
t.set_grad_enabled(False)

st.markdown(
r"""
# Semantic Similarity

Coming soon!

On this page, you'll be able to investigate semantic similarity, and how we're measuring it. You'll be able to type in a token, and find all its' semantic neighbours (as well as whether we're classifying it as a function word).

""", unsafe_allow_html=True)
This repo serves two purposes: 

1) An edited version of [TransformerLens](https://github.com/neelnanda-io/TransformerLens) with a couple of extra features (see below).
2) Hosting streamlit pages from https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages/blob/main/transformer_lens/rs/callum2/st_page/Home.py

See `transformer_lens/rs/arthurs_notebooks/example_notebook.py` for example usage.

## Setup:

This setup relies on using an SSH key to access Github. See [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and the associated links on that page (if you don't have an SSH key to begin with)

```bash
$ git clone git@github.com:callummcdougall/SERI-MATS-2023-Streamlit-pages.git
$ cd TransformerLens
$ pip install -e . # poetry install also should work
```

We stored some large files in git history and need clean them up; try `git clone --depth 1 git@github.com:callummcdougall/SERI-MATS-2023-Streamlit-pages.git` if git clone is lagging.

If you want to launch streamlit pages, run 

```python
pip install streamlit
cd transformer_lens/rs/callum2/st_page
streamlit run Home.py
```

## Difference from [the main branch of TransformerLens](https://github.com/neelnanda-io/TransformerLens)

1. We set the `ACCELERATE_DISABLE_RICH` environment variable in `transformer_lens/__init__.py` to `"1"` to stop an annoying reformatting of notebook error messages
2. We add the `qkv_normalized_input` hooks that can be optionally added to models

## [See the main TransformerLens README here](https://github.com/neelnanda-io/TransformerLens)

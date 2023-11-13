<h1>To view our Streamlit, visit https://copy-suppression.streamlit.app/ </h1>

<<<<<<< HEAD
(If you're interested in our research, please reach out! Our emails are `{cal.s.mcdougall, arthurconmy, thisiscodyr}@gmail.com`)

This repo serves two purposes: 
1) An edited version of [TransformerLens](https://github.com/neelnanda-io/TransformerLens) with a couple of extra features (see below).
2) Hosting streamlit pages from https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages/blob/main/transformer_lens/rs/callum2/st_page/Home.py

See `transformer_lens/rs/arthurs_notebooks/example_notebook.py` for example usage.

## Development Setup:
This setup assumes you're using an SSH key to access Github. See [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and the associated links on that page (if you don't have an SSH key to begin with)

```bash
$ git clone git@github.com:callummcdougall/SERI-MATS-2023-Streamlit-pages.git
$ cd TransformerLens
$ poetry install
```

`pip install -e` works too, though not using identical packages to `poetry.lock` plausibly could be a problem.

You need to have poetry installed; to do this run 
```
curl -sSL https://install.python-poetry.org | python3 -
```

and then either try to edit `PATH` manually or do `echo -e "$(cat ~/.bashrc)\nexport PATH=\"$HOME/.local/bin:\$PATH\"\n" > ~/.bashrc; source ~/.bashrc` to run through the poetry install tricks on a linux machine.

You should add requirements, e.g `einops`, via running `poetry add einops`.

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

---

# Guide to Experiments

* Surveying the direct effects of individual attention heads: `transformer_lens/rs/arthurs_notebooks/direct_effect_survey.py`
* (TODO: scan through the paper, ideally clean up the repo too)
* (TODO: write a better implementation of the learnable scale and bias vectors)

# Description of directories in `transformer_lens/rs/callum2`

*(Written by Callum)* These are the directories which I use to structure my own work.

## `ioi_and_bos`

This directory is for 2 small investigations:

1. How does the head manage to attend to BOS by default?

Conclusions - when you look at the cosine similarity of "residual stream vector before attn layer 10" and "query bias for head 10.7", it's very positive and in a very tight range for all tokens (between 0.45 and 0.47) whenever position is zero, and the same but very negative for all tokens whenever position isn't zero. So this isn't a function of BOS, it's a function of position. This has implications for how CSPA works; the query-side prediction has to overcome some threshold to actually activate the copy suppression mechanism.

2. What's the perpendicular component of the query, in IOI?

Conclusions - 

* Adding semantically similar tokens `"Mary"`, `"mary"` rather than just `" Mary"` doesn't seem to help.
* Found weak evidence that there's some kind of "indirect prediction", because when you take the perpendicular component and put them through the MLPs it does favour IO over S1 (but the MLPs don't have much impact in IOI so this effect isn't large anyway).

## `st_page`

Hosting all of the Streamlit pages. This isn't for generating any plots (at least I don't use it for that); it's exclusively for hosting pages & storing media files.

The pages are:

1. **OV and QK circuits** - you get to see what tokens are most attended to (QK circuit, prediction-attention) and what tokens are most suppressed (OV circuit). It's a nice way to highlight semantic similarity, and build intuition for how it works.
2. **Browse Examples** - the most important page. You get to investigate OWT examples, and see how all parts of the copy suppression mechanism works. You can:
    * See the loss change per token when you ablate, i.e. find the MIDS (tokens which the head is most helpful for).
    * See the logits pre and post-ablation, as well as the direct logit attribution for this head. *You can confirm that the head is pushing down tokens which appear in context, for most of the MIDS examples.*
    * Look at the attention patterns. *You can confirm that the head is attending to the tokens which it pushes down.*
    * Look at the logit lens before head 10.7. *You can confirm that the head is predicting precisely the words which it is attending to.*

## `cspa`

This is where I get the copy suppression-preserving ablation results. In other words, the stuff that's gonna be in section 3.3 of the paper (and that makes up one of the Streamlit pages).

It also adds to the HTML plots dictionary, for the "Browse Examples" Streamlit page.

## `ov_qk_circuits` 

This generates code for section 3.1, and generates the data for the following Streamlit pages:

1. OV and QK circuits

## `generate_st_html`

This is exclusively for generating the HTML figures that will be on the following Streamlit pages:

2. Browse Examples
3. Test Your Own Examples
=======
Optionally, if you want Jupyter Lab you can run `poetry run pip install jupyterlab` (to install in the same virtual environment), and then run with `poetry run jupyter lab`.

Then the library can be imported as `import transformer_lens`.

### Testing

If adding a feature, please add unit tests for it to the tests folder, and check that it hasn't broken anything major using the existing tests (install pytest and run it in the root TransformerLens/ directory).

#### Running the tests

- All tests via `make test`
- Unit tests only via `make unit-test`
- Acceptance tests only via `make acceptance-test`

### Formatting

This project uses `pycln`, `isort` and `black` for formatting, pull requests are checked in github actions.

- Format all files via `make format`
- Only check the formatting via `make check-format`

### Demos

If adding a feature, please add it to the demo notebook in the `demos` folder, and check that it works in the demo format. This can be tested by replacing `pip install git+https://github.com/neelnanda-io/TransformerLens.git` with `pip install git+https://github.com/<YOUR_USERNAME_HERE>/TransformerLens.git` in the demo notebook, and running it in a fresh environment.

## Citation
=======
<!-- Status Icons -->
[![Pypi](https://img.shields.io/pypi/v/transformer-lens?color=blue)](https://pypi.org/project/transformer-lens/)
![Pypi Total Downloads](https://img.shields.io/pepy/dt/transformer_lens?color=blue) ![PyPI -
License](https://img.shields.io/pypi/l/transformer_lens?color=blue) [![Release
CD](https://github.com/neelnanda-io/TransformerLens/actions/workflows/release.yml/badge.svg)](https://github.com/neelnanda-io/TransformerLens/actions/workflows/release.yml)
[![Tests
CD](https://github.com/neelnanda-io/TransformerLens/actions/workflows/checks.yml/badge.svg)](https://github.com/neelnanda-io/TransformerLens/actions/workflows/checks.yml)
[![Docs
CD](https://github.com/neelnanda-io/TransformerLens/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/neelnanda-io/TransformerLens/actions/workflows/gh-pages.yml)

A Library for Mechanistic Interpretability of Generative Language Models.

[![Read the Docs
Here](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://neelnanda-io.github.io/TransformerLens/)](https://neelnanda-io.github.io/TransformerLens/)

This is a library for doing [mechanistic
interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The
goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms
the model learned during training from its weights.

TransformerLens lets you load in 50+ different open source language models, and exposes the internal
activations of the model to you. You can cache any internal activation in the model, and add in
functions to edit, remove or replace these activations as the model runs.

~~ [OCTOBER SURVEY HERE](https://forms.gle/bw7U3PfioacDtFmT8) ~~

## Quick Start

### Install

```shell
pip install transformer_lens
```

### Use

```python
import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
```

## Key Tutorials

* [Introduction to the Library and Mech
  Interp](https://arena-ch1-transformers.streamlit.app/[1.2]_Intro_to_Mech_Interp)
* [Demo of Main TransformerLens Features](https://neelnanda.io/transformer-lens-demo)

## Gallery

Research done involving TransformerLens:

<!-- If you change this also change docs/source/content/gallery.md -->
* [Progress Measures for Grokking via Mechanistic
  Interpretability](https://arxiv.org/abs/2301.05217) (ICLR Spotlight, 2023) by Neel Nanda, Lawrence
  Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
* [Finding Neurons in a Haystack: Case Studies with Sparse
  Probing](https://arxiv.org/abs/2305.01610) by Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine
  Harvey, Dmitrii Troitskii, Dimitris Bertsimas
* [Towards Automated Circuit Discovery for Mechanistic
  Interpretability](https://arxiv.org/abs/2304.14997) by Arthur Conmy, Augustine N. Mavor-Parker,
  Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso
* [Actually, Othello-GPT Has A Linear Emergent World Representation](https://neelnanda.io/othello)
  by Neel Nanda
* [A circuit for Python docstrings in a 4-layer attention-only
  transformer](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)
  by Stefan Heimersheim and Jett Janiak
* [A Toy Model of Universality](https://arxiv.org/abs/2302.03025) (ICML, 2023) by Bilal Chughtai,
  Lawrence Chan, Neel Nanda
* [N2G: A Scalable Approach for Quantifying Interpretable Neuron Representations in Large Language
  Models](https://openreview.net/forum?id=ZB6bK6MTYq) (2023, ICLR Workshop RTML) by Alex Foote, Neel
  Nanda, Esben Kran, Ioannis Konstas, Fazl Barez
* [Eliciting Latent Predictions from Transformers with the Tuned
  Lens](https://arxiv.org/abs/2303.08112) by Nora Belrose, Zach Furman, Logan Smith, Danny Halawi,
  Igor Ostrovsky, Lev McKinney, Stella Biderman, Jacob Steinhardt

User contributed examples of the library being used in action:

* [Induction Heads Phase Change
  Replication](https://colab.research.google.com/github/ckkissane/induction-heads-transformer-lens/blob/main/Induction_Heads_Phase_Change.ipynb):
  A partial replication of [In-Context Learning and Induction
  Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  from Connor Kissane
* [Decision Transformer
  Interpretability](https://github.com/jbloomAus/DecisionTransformerInterpretability): A set of
  scripts for training decision transformers which uses transformer lens to view intermediate
  activations, perform attribution and ablations. A write up of the initial work can be found
  [here](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability).

Check out [our demos folder](https://github.com/neelnanda-io/TransformerLens/tree/main/demos) for
more examples of TransformerLens in practice

## Getting Started in Mechanistic Interpretability

Mechanistic interpretability is a very young and small field, and there are a _lot_ of open
problems. This means there's both a lot of low-hanging fruit, and that the bar for entry is low - if
you would like to help, please try working on one! The standard answer to "why has no one done this
yet" is just that there aren't enough people! Key resources:

* [A Guide to Getting Started in Mechanistic Interpretability](https://neelnanda.io/getting-started)
* [ARENA Mechanistic Interpretability Tutorials](https://arena-ch1-transformers.streamlit.app/) from
  Callum McDougall. A comprehensive practical introduction to mech interp, written in
  TransformerLens - full of snippets to copy and they come with exercises and solutions! Notable
  tutorials:
  * [Coding GPT-2 from
    scratch](https://arena-ch1-transformers.streamlit.app/[1.1]_Transformer_from_Scratch), with
    accompanying video tutorial from me ([1](https://neelnanda.io/transformer-tutorial)
    [2](https://neelnanda.io/transformer-tutorial-2)) - a good introduction to transformers
  * [Introduction to Mech Interp and
    TransformerLens](https://arena-ch1-transformers.streamlit.app/[1.2]_Intro_to_Mech_Interp): An
    introduction to TransformerLens and mech interp via studying induction heads. Covers the
    foundational concepts of the library
  * [Indirect Object
    Identification](https://arena-ch1-transformers.streamlit.app/[1.3]_Indirect_Object_Identification):
    a replication of interpretability in the wild, that covers standard techniques in mech interp
    such as [direct logit
    attribution](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=disz2gTx-jooAcR0a5r8e7LZ),
    [activation patching and path
    patching](https://www.lesswrong.com/posts/xh85KbTFhbCz7taD4/how-to-think-about-activation-patching)
* [Mech Interp Paper Reading List](https://neelnanda.io/paper-list)
* [200 Concrete Open Problems in Mechanistic
  Interpretability](https://neelnanda.io/concrete-open-problems)
* [A Comprehensive Mechanistic Interpretability Explainer](https://neelnanda.io/glossary): To look
  up all the jargon and unfamiliar terms you're going to come across!
* [Neel Nanda's Youtube channel](https://www.youtube.com/channel/UCBMJ0D-omcRay8dh4QT0doQ): A range
  of mech interp video content, including [paper
  walkthroughs](https://www.youtube.com/watch?v=KV5gbOmHbjU&list=PL7m7hLIqA0hpsJYYhlt1WbHHgdfRLM2eY&index=1),
  and [walkthroughs of doing
  research](https://www.youtube.com/watch?v=yo4QvDn-vsU&list=PL7m7hLIqA0hr4dVOgjNwP2zjQGVHKeB7T)

## Support & Community

[![Contributing
Guide](https://img.shields.io/badge/-Contributing%20Guide-blue?style=for-the-badge&logo=GitHub&logoColor=white)](https://neelnanda-io.github.io/TransformerLens/content/contributing.html)

If you have issues, questions, feature requests or bug reports, please search the issues to check if
it's already been answered, and if not please raise an issue!

You're also welcome to join the open source mech interp community on
[Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-1qosyh8g3-9bF3gamhLNJiqCL_QqLFrA).
Please use issues for concrete discussions about the package, and Slack for higher bandwidth
discussions about eg supporting important new use cases, or if you want to make substantial
contributions to the library and want a maintainer's opinion. We'd also love for you to come and
share your projects on the Slack!

## Credits

This library was created by **[Neel Nanda](https://neelnanda.io)** and is maintained by **Joseph
Bloom**.

The core features of TransformerLens were heavily inspired by the interface to [Anthropic's
excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson
Elhage and Chris Olah for building Garcon and showing the value of good infrastructure for enabling
exploratory research!

### Creator's Note (Neel Nanda)

I (Neel Nanda) used to work for the [Anthropic interpretability team](transformer-circuits.pub), and
I wrote this library because after I left and tried doing independent research, I got extremely
frustrated by the state of open source tooling. There's a lot of excellent infrastructure like
HuggingFace and DeepSpeed to _use_ or _train_ models, but very little to dig into their internals
and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get
into the field even if you don't work at an industry org with real infrastructure! One of the great
things about mechanistic interpretability is that you don't need large models or tons of compute.
There are lots of important open problems that can be solved with a small model in a Colab notebook!

### Citation
>>>>>>> felix/support-mistral-7B

Please cite us with :

<<<<<<< HEAD
```
@article{copy_suppression,
  title={Copy Suppression: Comprehensively Understanding an Attention Head},
  author={McDougall, Callum and Conmy, Arthur and Rushing, Cody and McGrath, Thomas and Nanda, Neel},
  journal={arXiv preprint},
  year={2023},
}
```

(arXiv should be out soon!)
=======
```BibTeX
@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Neel Nanda and Joseph Bloom},
    year = {2022},
    howpublished = {\url{https://github.com/neelnanda-io/TransformerLens}},
}
```
>>>>>>> felix/support-mistral-7B

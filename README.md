# Atari Representation Learning Ablation Study
NeurIPS Reproducibility Challenge 2019 for
[Unsupervised State Representation Learning in Atari](https://arxiv.org/pdf/1906.08226.pdf)

# This repository contains

Some code we used to perform our ablation study such as custom trainers and encoders that extend those
found in the original article's codebase, along with training scripts. To see the original codebase
see https://github.com/mila-iqia/atari-representation-learning.git

To explore all of our training runs and results go to https://app.wandb.ai/neurips-challenge/atari-ari/
in total we used 17 days of GPU time spread across K80 GPUs and occasionally P100 GPUs provided by Google Colab.

We also contributed to the main MILA codebase with some changes we needed for our work, see this pull request for details: https://github.com/mila-iqia/atari-representation-learning/pull/48 

# Setup

```
pip install -r requirements.txt
pip install git+git://github.com/openai/baselines
pip install git+git://github.com/ankeshanand/pytorch-a2c-ppo-acktr-gail
pip install git+git://github.com/mila-iqia/atari-representation-learning.git
```

#! /usr/bin/env python
"""
This script should reproduce our best submission on Kaggle, using an already
trained CNN architecture for twitter sentiment analysis.

Our model is (at least) 1.6 GB, so we didn't attach it. Instead, you
can download it using the `get_data.py` script (which this script calls)

For the training, see the file `train_CNN.py`, and for evaluation, `eval_CNN.py`

Our code is an adaptation from the open source (Apache license) software at
https://github.com/dennybritz/cnn-text-classification-tf

Authors: András Ecker, Valentin Vasiliu, Ciprian I. Tomoiagă
"""


# Data preparation:
# ==================================================
import get_data
get_data.main()

# Model evaluation:
# ==================================================
import eval_CNN

## Library
# base
import argparse
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')

# data manipulation
import numpy as np
import pandas as pd

# tools
import torch

# user-made tools
from utils.data import get_new_data
from utils.models import pred_model

def define_argparser():
    parser = argparse.ArgumentParser()

    # Pretrain model
    parser.add_argument('--pretrain-model', type=str, default='BioBERT',
                        choices=["BERT", "BioBERT", "BioBERT_pubmed", "SciBERT", "AlBERT", "causality"],
                        help='default BioBERT. Arg:pretrain_model can be added after modifying pretrain_dcit in line 80')

    # Settings
    parser.add_argument('--max-len', type=float, default=128,
                        help='default 128. Set maximum token length for input')
    parser.add_argument('--num-class', type=int, default=4,
                        help='default 4. Value for number of labels in dataset.')
    
    # Directory
    parser.add_argument('--output-path', type=str, default='/results/',
                        help='default directory was set to models folder. \
                        Write down another path if you want to save model in different directory')
    parser.add_argument('--data-path', type=str, default='/data/sample_new_sentences.csv',
                        help='default data directory for testing. \
                        Write down another path if you want to load your own data')
    parser.add_argument('--model-path', type=str, default='/pytorch_model.bin',
                        help='default saved model directory. \
                        Write down another path if you want to load your own model')

    config = parser.parse_args()
    return config

def update_config(config):
    # Set directory
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    config.output_dir_path = current_dir + config.output_path + config.pretrain_model  # output directory
    config.data_file_path = current_dir + config.data_path  # data directory
    config.model_dir_path = config.output_dir_path + config.model_path  # model directory

    # Set pretrain model path
    pretrain_dict = {
        "BERT": "bert-base-cased",
        "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
        "BioBERT_pubmed": "monologg/biobert_v1.1_pubmed",
        "SciBERT": "allenai/scibert_scivocab_cased",
        "AlBERT": "tals/albert-xlarge-vitaminc-mnli",
        "causality": "adamnik/bert-causality-baseline"
    }
    config.pretrain_path = pretrain_dict[config.pretrain_model]

    # set device
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return config


def main(config):
    args = update_config(config)

    # Test dataset
    df, inputs = get_new_data(args)
    labels, probs = pred_model(inputs, args)

    df['pred_labels'] = labels
    df['pred_probs'] = probs
    df.to_csv(args.output_dir_path+'pred.csv', index=False, float_format="%.3f")


if __name__ == '__main__':
    config = define_argparser()
    main(config)
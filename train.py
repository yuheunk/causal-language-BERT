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
from utils.data import get_encoded_dataset
from utils.models import train_model, eval_metrics

def define_argparser():
    parser = argparse.ArgumentParser()

    # Train
    parser.add_argument('--train-method', type=str, default='trainAll',
                        choices=['trainAll', 'Kfold'],
                        help='default trainAll.')
    parser.add_argument('--pretrain-model', type=str, default='BioBERT',
                        choices=["BERT", "BioBERT", "BioBERT_pubmed", "SciBERT", "AlBERT", "causality"],
                        help='default BioBERT. Arg:pretrain_model can be added after modifying pretrain_dcit in line 80')

    # Model settings
    parser.add_argument('--seed', type=int, default=0,
                        help='fixed random seed for reproducibility. default 0')
    parser.add_argument('--epochs', type=int, default=5,
                        help='default 5')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='default 32')
    parser.add_argument('--max-len', type=float, default=128,
                        help='default 128. Set maximum token length for input')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='default 2e-5. Set learning rate for AdamW optimizer.')
    parser.add_argument('--K', type=int, default=5,
                        help='default 5. Value for nubmer of cross-validations.')
    parser.add_argument('--num-class', type=int, default=4,
                        help='default 4. Value for number of labels in dataset.')
    
    # Data directory
    parser.add_argument('--output-path', type=str, default='/results/',
                        help='default directory was set to models folder. \
                        Write down another path if you want to save model in different directory')
    parser.add_argument('--data-path', type=str, default='/data/pubmed_causal_language_use.csv',
                        help='default data directory for training was set. \
                        Write down another path if you want to load your own data')

    config = parser.parse_args()
    return config

def update_config(config):
    # Set directory
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    config.output_dir_path = current_dir + config.output_path + config.pretrain_model  # output directory
    config.data_file_path = current_dir + config.data_path  # data directory

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
    return config


def main(config):
    args = update_config(config)

    # for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # in case of using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model
    ## Train all data
    if args.train_method=='trainAll':
        train_dataset, test_dataset = get_encoded_dataset(args)
        trainer = train_model(train_dataset=train_dataset, \
                            test_dataset=test_dataset, \
                            config=args, \
                            model_file_to_save = args.output_dir_path
                            )
        
        trainer.save_model(args.output_dir_path)
        print(f'\n- model saved to: {args.output_dir_path}\n')

    ## Kfold uses all fold
    elif args.train_method=='Kfold':
        results = []
        for fold in range(args.K):
            train_dataset, test_dataset = get_encoded_dataset(args, 
                                                              fold=fold  # add fold
                                                              )
            
            output_dir_path_K = args.output_dir_path+f'_K({str(args.K)})_fold{str(fold)}'  # change model save directory

            trainer = train_model(train_dataset=train_dataset, \
                                test_dataset=test_dataset, \
                                config=args, \
                                model_file_to_save=output_dir_path_K # change output dir path
                                )
            
            items = eval_metrics(trainer)
            results.append(items)
        
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
       
        df_2.to_csv(current_dir + args.output_dir_path + "test_metrics.csv", index=False, float_format="%.3f")
        

if __name__ == '__main__':
    config = define_argparser()
    main(config)
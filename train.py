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
    parser.add_argument('--pretrain-model', type=str, default='BioBERT',
                        choices=["BERT", "BioBERT", "BioBERT_pubmed", "SciBERT", "AlBERT", "causality"],
                        help='default BioBERT. Arg:pretrain_model can be added after modifying pretrain_dcit in line 80')

    parser.add_argument('--train-method', type=str, default='simple',
                        choices=['simple', 'full', 'Kfold'],
                        help='default simple.')

    parser.add_argument('--seed', type=int, default=0,
                        help='fixed random seed for reproducibility. default 0')

    parser.add_argument('--epochs', type=int, default=5,
                        help='default 5')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='default 32')

    parser.add_argument('--max-len', type=float, default=128,
                        help='default 128. Set maximum token length for input')

    parser.add_argument('--lr', type=float, default=2e-5,
                        help='default 2e-5. Set learning rate for AdamW optimizer.')

    parser.add_argument('--K', type=int, default=5,
                        help='default 5. Value for nubmer of cross-validations.')
   
    parser.add_argument('--output-dir-path', type=str, default='/results/',
                        help='default directory was set to models folder. \
                        Write down another path if you want to save model in different directory')

    parser.add_argument('--data_file_path', type=str, default='/data/pubmed_causal_language_use.csv',
                        help='default data directory for training was set. \
                        Write down another path if you want to load your own data')

    config = parser.parse_args()
    return config

def main(config):
    # for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)  # in case of using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set directory
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    # Output directory
    output_dir_path = current_dir + config.output_dir_path + config.pretrain_model
    data_file_path = current_dir + config.data_file_path

    # Set pretrain model path
    pretrain_dict = {
        "BERT": "bert-base-cased",
        "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
        "BioBERT_pubmed": "monologg/biobert_v1.1_pubmed",
        "SciBERT": "allenai/scibert_scivocab_cased",
        "AlBERT": "tals/albert-xlarge-vitaminc-mnli",
        "causality": "adamnik/bert-causality-baseline"
    }
    pretrain_path = pretrain_dict[config.pretrain_model]
    
    # Train model
    ## simple only uses the first fold
    if config.train_method=='trainAll':
        train_dataset, test_dataset = get_encoded_dataset(data_dir=data_file_path,
                                                        pretrain_model=pretrain_path,
                                                        K=config.K,
                                                        length=config.max_len,
                                                        random_state=config.seed)
        trainer = train_model(train_dataset=train_dataset, \
                            test_dataset=test_dataset, \
                            model_file_to_save=output_dir_path, \
                            epochs=config.epochs, \
                            lr=config.lr, \
                            batch_size=config.batch_size, \
                            pretrain_model=pretrain_path)
        
        trainer.save_model(output_dir_path)
        print(f'\n- model saved to: {model_file_to_save}\n')

    ## Kfold uses all fold
    elif config.train_method=='Kfold':
        results = []
        for fold in range(config.K):
            train_dataset, test_dataset = get_encoded_dataset(data_dir=data_file_path,
                                                        pretrain_model=pretrain_path,
                                                        K=config.K,
                                                        length=config.max_len,
                                                        random_state=config.seed,
                                                        fold=fold  # add fold
                                                        )
            
            output_dir_path_K = output_dir_path+f'_K({str(config.K)})_fold{str(fold)}'  # change model save directory

            trainer = train_model(train_dataset=train_dataset, \
                                test_dataset=test_dataset, \
                                model_file_to_save=output_dir_path_K, \
                                epochs=config.epochs, \
                                lr=config.lr, \
                                batch_size=config.batch_size, \
                                pretrain_model=pretrain_path)
            
            items = eval_metrics(trainer)
            results.append(items)
        
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
       
        df_2.to_csv(current_dir + config.output_dir_path + "test_metrics.csv", index=False, float_format="%.3f")
    
    # elif config.train_method=='trainAll':
    #     pass

        

if __name__ == '__main__':
    config = define_argparser()
    main(config)
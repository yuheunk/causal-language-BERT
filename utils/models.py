import os
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import transformers
from transformers import TrainingArguments, Trainer, BertForSequenceClassification

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.data import read_data

#!# Should be modified
DATA_DIR = "/home/ykim72/Lantis/causal-language-BERT/data/pubmed_causal_language_use.csv"



class MyTrainer(Trainer):
    # def __init__(self, data_dir, model, args, train_dataset, eval_dataset, compute_metrics) -> None:
    #     super(MyTrainer, self).__init__(
    #         model=model,
    #         args=args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         compute_metrics=compute_metrics
    #     )
    #     self.data_dir = data_dir

    def compute_loss(self, model, inputs, return_outputs=False): # data_dir,
        df = read_data(DATA_DIR)
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        class_weights_ = [x for x in compute_class_weight("balanced", classes=range(len(set(df.label))), y=df.label)]
        class_weights = torch.FloatTensor(class_weights_).cuda()
        loss_func = CrossEntropyLoss(weight=torch.tensor(class_weights))
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)
        return output.metrics, output.predictions, output.label_ids
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds))
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(train_dataset, test_dataset, args, model_file_to_save):
    # Data directory
    data_file_path = '..' + args.data_file_path

    model = BertForSequenceClassification.from_pretrained(args.pretrain_path, num_labels=args.num_class)
    model.to(args.device)
    model.train()
    print('ready')
    
    training_args = TrainingArguments(
        output_dir=model_file_to_save,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = MyTrainer(
        # data_dir=data_file_path, #!#
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    print('### START TRAINING ###')
    print(f'pretrain model = {args.pretrain_path}')
    trainer.train()
    print('### TRAINING DONE ###')

    trainer.save_model(model_file_to_save)
    print(f'\n- model saved to: {model_file_to_save}\n')
    return trainer

def eval_metrics(trainer):
    print('### GET METRICS ###')
    metrics = trainer.evaluate()
    metric, preds, labels = metrics[0], metrics[1].argmax(-1), metrics[2]
    items = {
            'Acc': metric['eval_accuracy'], 
            'P': metric['eval_precision'], 
            'R': metric['eval_recall'], 
            'F1': metric['eval_f1']
            }

    report = classification_report(labels, preds, output_dict=True)

    for cls in np.unique(labels):
        tmp = ['precision', 'recall', 'f1-score']
        for i, scoring in enumerate('P R F1'.split()):
            items['{}_{}'.format(scoring, cls)] = report[str(cls)][tmp[i]]

    trainer.save_metrics("eval", metric)
    print(f'\n- metric saved.')
    return items

def load_model(args):
    model = BertForSequenceClassification.from_pretrained(args.pretrain_path, num_labels=args.num_class)
    model.load_state_dict(torch.load(args.model_dir_path))
    return model

def pred_model(inputs, args):
    model = load_model(args)
    model.to(args.device)
    model.eval()
    print('ready')
    
    with torch.no_grad():
        logits = model(**inputs.to(args.device)).logits
    logits = logits.detach().cpu()
    labels = np.argmax(logits.numpy(), axis=1)
    probs = np.argmax(F.softmax(logits, dim=-1), axis=1)
    return labels, probs
import os
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
import transformers
from transformers import TrainingArguments, Trainer, BertForSequenceClassification

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.data import read_data

# Should be modified
DATA_DIR = "/home/ykim72/Lantis/causal-language-use-in-science/data/pubmed_causal_language_use.csv"
label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}
NUM_CLASSES = len(label_name)

###
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
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

def train_model(train_dataset, test_dataset, model_file_to_save, epochs, lr, batch_size, pretrain_model="dmis-lab/biobert-base-cased-v1.2"):
    model = BertForSequenceClassification.from_pretrained(pretrain_model, num_labels=NUM_CLASSES)
    model.to(device)
    model.train()
    print('ready')
    
    training_args = TrainingArguments(
        output_dir=model_file_to_save,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy="epoch"
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    print('### START TRAINING ###')
    print(f'pretrain model = {pretrain_model}')
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
    
import torch
from torch.utils.data import DataLoader
import numpy as np

from config import device, batch_size, model_params, embed_params, encoder_params, data_params, training_params, paths, ensemble_models

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models import RNNClassifier

from modules.layers.embeddings import ELMo, LookUp, ELMoForManyLangs
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data_file, load_data_file_without_split, split_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

from utils.data import remap_data_to_2, remap_data_to_3, remap_data_to_3_from_text

print('Loading dataset...')
preprocessing = Preprocessing()

data = load_data_file_without_split(**data_params)
data = remap_data_to_3_from_text(data)


train_data, valid_data, test_data = split_data(data)
x_column, y_column = data_params['x_column'], data_params['y_column']

train_set = ClassificationDataset(train_data[:, x_column], train_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set = ClassificationDataset(valid_data[:, x_column], valid_data[:, y_column], preprocessing=preprocessing.process_text)
test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)


train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)

print('Creating model...')

embeddings = ELMoForManyLangs(**embed_params)
model = RNNClassifier(embeddings, encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.NLLLoss(weight=weights)

trainer = ClassificationTrainer(model, criterion, optimizer, device)

print('Training...')
predictions = []
losses = []
gold_labels = test_set.labels.astype(int)


for model_name in ensemble_models:
    trainer.model = torch.load('checkpoints/' + model_name)

    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader)
    predictions.append(model_predictions)
    losses.append(test_loss)

    print('----------------------------------------------------Test results----------------------------------------------------')
    print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
    print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
    print('| Macro F1: {} | Micro F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro')))
    print('--------------------------------------------------------------------------------------------------------------------')


print('Sum ensemble')

sum_predictions = np.stack(predictions).sum(axis=0)
predicted = np.argmax(sum_predictions, 1)

print('----------------------------------------------------Test results----------------------------------------------------')
print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro')))
print('--------------------------------------------------------------------------------------------------------------------')

print('Voting ensemble')

leaderboard = np.zeros(predictions[0].shape)
for index, prediction in enumerate(predictions):
    for j in range(prediction.shape[0]):
        leaderboard[j][prediction[j].argmax()] += 1

predicted = np.argmax(leaderboard, axis=1)

print('----------------------------------------------------Test results----------------------------------------------------')
print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
print('| Macro F1: {} | Micro F1: {} |'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro')))
print('--------------------------------------------------------------------------------------------------------------------')

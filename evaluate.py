import torch
from torch.utils.data import DataLoader

from config import device, batch_size, model_params, embed_params, encoder_params, data_params, training_params, paths

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from modules.layers.embeddings import ELMo, LookUp, ELMoForManyLangs
from modules.common import ConcatVocabulary
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data_file, load_data_file_without_split, split_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

from utils.model import RNNClassifier
from utils.data import remap_data_to_2, remap_data_to_3, remap_data_to_3_from_text, save_predictions

print('Loading dataset...')
preprocessing = Preprocessing()

data = load_data_file_without_split(**data_params)

# data = remap_data_to_3(data)
data = remap_data_to_3_from_text(data)

print(get_histogram_data(data[:, 2]))

train_data, valid_data, test_data = split_data(data)
x_column, y_column = data_params['x_column'], data_params['y_column']

test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)

print('Creating model...')

embeddings = ELMoForManyLangs(**embed_params)

model = RNNClassifier(embeddings, encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.NLLLoss(weight=weights)

trainer = ClassificationTrainer(model, criterion, optimizer, device)

print('Training...')
best_macro_f1 = None
gold_labels = test_set.labels.astype(int)

for epoch in range(training_params['n_epochs']):
    trainer.model = torch.load('checkpoints/best_valid_f1_model')

    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader)
    print('----------------------------------------------------Test results----------------------------------------------------')
    print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
    print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
    print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
    print('| Macro F1: {} | Micro F1: {}|'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro')))
    print('--------------------------------------------------------------------------------------------------------------------')

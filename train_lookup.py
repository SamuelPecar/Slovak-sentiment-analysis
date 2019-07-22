import torch
from torch.utils.data import DataLoader

from config import device, batch_size, model_params, embed_params, encoder_params, data_params, training_params, paths

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from modules.layers.embeddings import LookUp, ELMoForManyLangs
from modules.common import ConcatVocabulary
from modules.common.preprocessing import Preprocessing
from modules.common.utils import class_weigths
from modules.common.dataloading import load_data_file, load_data_file_without_split, split_data, collate_fn_cf
from modules.trainers import ClassificationTrainer
from modules.datasets.classification_dataset import ClassificationDataset

from utils.model import RNNClassifier
from utils.data import remap_data_to_3, remap_data_to_3_from_text, load_vectors

# vectors = load_vectors('./data/prim-6.1-public-all.shuffled.300cbow.bin')
# vectors.save_word2vec_format('./data/prim-6.1-public-all.shuffled.300cbow.txt', binary=False)

def get_histogram_data(set):
    histogram = {}

    for item in set:
        if item in histogram:
            histogram[item] += 1
        else:
            histogram[item] = 1
    return histogram


print('Loading dataset...')
preprocessing = Preprocessing()

data = load_data_file_without_split(**data_params)
# data = remap_data_to_3(data)
data = remap_data_to_3_from_text(data)

print(get_histogram_data(data[:, 2]))

train_data, valid_data, test_data = split_data(data)
x_column, y_column = data_params['x_column'], data_params['y_column']

train_set = ClassificationDataset(train_data[:, x_column], train_data[:, y_column], preprocessing=preprocessing.process_text)
valid_set = ClassificationDataset(valid_data[:, x_column], valid_data[:, y_column], preprocessing=preprocessing.process_text)
test_set = ClassificationDataset(test_data[:, x_column], test_data[:, y_column], preprocessing=preprocessing.process_text)

vocab = ConcatVocabulary([train_set.vocab, valid_set.vocab, test_set.vocab])

train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
valid_loader = DataLoader(valid_set, batch_size, shuffle=True, collate_fn=collate_fn_cf)
test_loader = DataLoader(test_set, batch_size, collate_fn=collate_fn_cf)

print(get_histogram_data(train_set.labels))
print(get_histogram_data(valid_set.labels))
print(get_histogram_data(test_set.labels))

print('Creating model...')

# embeddings = LookUp(vocab=vocab, file_name='data/cc.sk.300.vec', type='word2vec', trainable=True, embedding_dim=300, embedding_dropout=.0)
embeddings = LookUp(vocab=vocab, file_name='data/prim-6.1-public-all.shuffled.300cbow.txt', type='word2vec', trainable=True, embedding_dim=300, embedding_dropout=.0)
# embeddings = LookUp(vocab=vocab, file_name=None, trainable=True, embedding_dim=300, embedding_dropout=.0)

model = RNNClassifier(embeddings, encoder_params, **model_params).to(device)

optimizer = torch.optim.Adam(model.parameters())

weights = class_weigths(train_set.labels).to(device)
criterion = torch.nn.NLLLoss(weight=weights)

trainer = ClassificationTrainer(model, criterion, optimizer, device)

print('Training...')
best_macro_f1 = None
gold_labels = test_set.labels.astype(int)

for epoch in range(training_params['n_epochs']):

    train_loss = trainer.train_model(train_loader)

    valid_loss, predicted, model_predictions, labels = trainer.evaluate_model(valid_loader)

    print('| Epoch: {} | Train Loss: {:2.5f} | Val. Loss: {:2.5f} | Val. Acc: {:2.5f} | Val. Macro F1: {:2.5f} | Val. Micro F1: {:2.5f} |'
          .format(epoch + 1, train_loss, valid_loss, accuracy_score(labels, predicted),
                  f1_score(labels, predicted, average='macro'), f1_score(labels, predicted, average='micro')))

    macro_f1 = f1_score(labels, predicted, average='macro')

    test_loss, predicted, model_predictions, labels = trainer.evaluate_model(test_loader)
    print('----------------------------------------------------Test results----------------------------------------------------')
    print('| Loss: {} | Acc: {}% |'.format(test_loss, accuracy_score(labels, predicted)))
    print('| Macro Precision: {} | Micro Precision: {} |'.format(precision_score(gold_labels, predicted, average='macro'), precision_score(gold_labels, predicted, average='micro')))
    print('| Macro Recall: {} | Micro Recall: {} |'.format(recall_score(gold_labels, predicted, average='macro'), recall_score(gold_labels, predicted, average='micro')))
    print('| Macro F1: {} | Micro F1: {}|'.format(f1_score(gold_labels, predicted, average='macro'), f1_score(gold_labels, predicted, average='micro')))
    print('--------------------------------------------------------------------------------------------------------------------')

    if not best_macro_f1 or macro_f1 > best_macro_f1:
        print('saving...')
        best_macro_f1 = macro_f1
        torch.save(model, paths['f1_score']['model_path'])

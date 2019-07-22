import torch
import torch.nn as nn

from modules.common.utils import sort_by_lengths

from modules.layers.encoders.rnn_encoder import RNNEncoder
from modules.layers.attention import SelfAttention


class RNNClassifier(nn.Module):
    def __init__(self, embeddings, encoder_params, output_dim=2, dropout=0., attention=True, **kwargs):
        super(RNNClassifier, self).__init__()

        self.embeddings = embeddings
        self.encoder = RNNEncoder(input_size=self.embeddings.embedding_dim, **encoder_params)

        if attention:
            self.attention = SelfAttention(attention_size=self.encoder.feature_size, dropout=dropout)
        else:
            self.attention = None

        self.hidden2out = nn.Linear(self.encoder.feature_size, output_dim)

    def forward(self, inputs, mask, hidden, lengths):
        sorted_lengths, sort, unsort = sort_by_lengths(lengths)
        batch_size = len(inputs)

        embedded, mask = self.embeddings(inputs)
        output_encoder, (ht, ct) = self.encoder(sort(embedded), hidden=None, mask=sort(mask), lengths=sorted_lengths)

        if self.attention:
            representations, attentions = self.attention(output_encoder, mask=sort(mask), lengths=sorted_lengths)
        else:
            representations = ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        output = self.hidden2out(unsort(representations))

        return output, hidden

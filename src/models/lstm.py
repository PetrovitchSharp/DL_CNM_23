from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Перевод символов в индексы (для использования в nn.Embedding)
characters = [' ', "'", '0', '1', '2',
              '3', '4', '5', '6', '7',
              '8', '9', 'a', 'b', 'c',
              'd', 'e', 'f', 'g', 'h',
              'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r',
              's', 't', 'u', 'v', 'w',
              'x', 'y', 'z']

char_dict = {v: k + 1 for k, v in enumerate(characters)}


def encode_string(s: str, char_dict=char_dict) -> torch.LongTensor:
    unknown = len(char_dict) + 1
    idxs = torch.LongTensor([char_dict.get(c, unknown) for c in s])
    return idxs


class LSTMNetwork(nn.Module):
    def __init__(self,
                 chars: Dict[str, int] = char_dict,
                 emb_dim: int = 20,
                 hidden_size: int = 100,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 proj_size: int = 0):
        super(LSTMNetwork, self).__init__()
        self.chars = chars
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(len(self.chars) + 2, emb_dim)
        self.LSTM = nn.LSTM(input_size=emb_dim,
                            hidden_size=self.hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout,
                            proj_size=proj_size)

    def emb_packed_sequence(self, packed_sequence):
        return torch.nn.utils.rnn.PackedSequence(
            self.emb(packed_sequence.data),
            packed_sequence.batch_sizes
            )

    def forward(self, x: List[str]):
        # Кодируем текст в соответствии со словарем символов
        encoded = [encode_string(el) for el in x]
        text_lengths = np.array([len(el) for el in x])

        # добавляем паддинг + упаковываем
        padded_encoded = torch.nn.utils.rnn.pad_sequence(encoded).to(device)
        packed_encoded = torch.nn.utils.rnn.pack_padded_sequence(
            padded_encoded, text_lengths)

        # пропускаем через слой с эмбеддингами и LSTM
        packed_emb = self.emb_packed_sequence(packed_encoded)
        packed_lstm_output, _ = self.LSTM(packed_emb)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_output)

        # собираем представление текста с учетом длины исходного текста
        fwd_state = output[text_lengths - 1,
                           range(len(text_lengths)),
                           :self.hidden_size]
        rev_state = output[0, :, :self.hidden_size]
        output = torch.cat((fwd_state, rev_state), 1)

        return output

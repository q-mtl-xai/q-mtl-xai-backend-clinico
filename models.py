
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights


class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, cnn_filters=(32, 64, 128),
                 lstm_hidden=128, embed_dim=128):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for n_filters in cnn_filters:
            layers.append(nn.Conv1d(prev_ch, n_filters, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(n_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            prev_ch = n_filters
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionPooling(lstm_hidden * 2)
        self.fc = nn.Linear(lstm_hidden * 2, embed_dim)

    def forward(self, x, return_attention=False):
        h = self.cnn(x)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        pooled, weights = self.attention(h)
        z = self.fc(pooled)
        if return_attention:
            return z, weights
        return z


class MultiTaskEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_latido_classes=4,
                 n_ventana_classes=4, n_isc_classes=2):
        super().__init__()
        self.shared_encoder = SharedEncoder(in_channels=1, embed_dim=embed_dim)
        self.isc_encoder = SharedEncoder(in_channels=12, embed_dim=embed_dim)
        self.head_latido = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_latido_classes),
        )
        self.head_ventana = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_ventana_classes),
        )
        self.head_isc = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_isc_classes),
        )

    def forward(self, x, task='latido', return_attention=False):
        if task == 'isc':
            z, w = self.isc_encoder(x, return_attention=True)
        else:
            z, w = self.shared_encoder(x, return_attention=True)

        if task == 'latido':
            out = self.head_latido(z)
        elif task == 'ventana':
            out = self.head_ventana(z)
        elif task == 'isc':
            out = self.head_isc(z)

        if return_attention:
            return out, w
        return out

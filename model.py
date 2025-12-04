# model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class PriceEncoder(nn.Module):
    """
    CNN + BiLSTM encoder over a sequence of OHLCV features.
    Input: (batch, seq_len, num_features)
    Output: (batch, price_hidden_dim)
    """
    def __init__(self, num_features: int, cnn_channels: int = 64,
                 lstm_hidden: int = 128, lstm_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features

        # Conv over time dimension
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(cnn_channels)

        # BiLSTM over conv output
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.output_dim = 2 * lstm_hidden

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        """
        price_seq: (batch, seq_len, num_features)
        """
        # (B, F, T)
        x = price_seq.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # (B, T, C)
        x = x.permute(0, 2, 1)

        # BiLSTM
        x, (h_n, c_n) = self.lstm(x)
        # Concatenate last hidden states from both directions
        # h_n: (num_layers*2, B, H)
        h_fwd = h_n[-2, :, :]
        h_bwd = h_n[-1, :, :]
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2H)
        h = self.dropout(h)
        return h


class HybridFinBERTModel(nn.Module):
    """
    Hybrid model:
      - FinBERT over headline text
      - CNN+BiLSTM over OHLCV price history
      - Fused MLP to predict next-day return (regression)
    """
    def __init__(
        self,
        finbert_model_name: str = "ProsusAI/finbert",
        price_num_features: int = 7,  # [open, high, low, close, adj_close, volume, maybe others]
        price_hidden_dim: int = 256,
        fusion_hidden_dim: int = 512,
        freeze_finbert: bool = True,
    ):
        super().__init__()

        # --- Text encoder (FinBERT) ---
        self.finbert = AutoModel.from_pretrained(finbert_model_name)
        finbert_hidden = self.finbert.config.hidden_size

        if freeze_finbert:
            for p in self.finbert.parameters():
                p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(finbert_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # --- Price encoder ---
        self.price_encoder = PriceEncoder(
            num_features=price_num_features,
            cnn_channels=64,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.1,
        )

        # --- Fusion head ---
        fusion_input_dim = 256 + self.price_encoder.output_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # predict ret
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids, attention_mask: for FinBERT
        price_seq: (batch, seq_len, num_features)
        """
        # Text part (use CLS)
        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]  # (B, H)
        text_emb = self.text_proj(cls)

        # Price part
        price_emb = self.price_encoder(price_seq)

        # Fuse
        fused = torch.cat([text_emb, price_emb], dim=-1)
        pred = self.fusion_mlp(fused).squeeze(-1)  # (B,)
        return pred

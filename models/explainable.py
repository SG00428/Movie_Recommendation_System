# models/explainable_model.py
import torch
import torch.nn as nn

class AttentionNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(AttentionNCF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)

        self.attention = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        features = torch.cat([u, i], dim=1)
        attn_weights = self.attention(features)
        out = self.fc(features * attn_weights)
        return out.squeeze(), attn_weights

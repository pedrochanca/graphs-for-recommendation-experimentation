import torch
import torch.nn as nn


class SimpleNCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32):
        """
        emb_dim = 32

        item and user embeddings get concatenated, resulting in an embedding with 64d.
        """

        super().__init__()

        # learnable parameters - user and item embedding matrices
        # user embedding matrix size = n_users x emb_dim
        # item embedding matrix size = n_items x emb_dim
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # single linear layer: 64 -> 1
        self.output = nn.Linear(2 * emb_dim, 1)

    def forward(self, user_ids, item_ids):
        """
        Zero hidden layers.

        All it does: for the 64d concatenated embedding, it outputs a single value that
        passes through a linear layer.
        """
        u_emb = self.user_embedding(user_ids)  # size: [batch, 32]
        i_emb = self.item_embedding(item_ids)  # size: [batch, 32]

        x = torch.cat([u_emb, i_emb], dim=1)
        output = self.output(x)

        return output


class DeepNCF(nn.Module):
    """
    nn.Module - when we call the class, it automatically executes the forward function
    """

    def __init__(self, n_users, n_items, emb_dim=32):
        """
        (original paper)
        layers (3): 64d - 32d - 16d (2 hidden layers + output layer / last hidden layer)
        """
        return None

    def forward(self, user_ids, item_ids):
        return None

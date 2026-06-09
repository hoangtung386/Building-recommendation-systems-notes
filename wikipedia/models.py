import torch
import torch.nn as nn


class Glove(nn.Module):
    """A simple embedding model based on GloVe.
       https://nlp.stanford.edu/projects/glove/
    """

    def __init__(self, num_embeddings: int = 1024, features: int = 64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self._token_embedding = nn.Embedding(num_embeddings, features)
        self._bias = nn.Embedding(num_embeddings, 1)
        nn.init.zeros_(self._bias.weight)

    def forward(self, inputs):
        """Calculates the approximate log count between tokens 1 and 2.

        Args:
          A batch of (token1, token2) integers implementing co-occurrence.

        Returns:
          Approximate log count between x and y.
        """
        token1, token2 = inputs
        embed1 = self._token_embedding(token1)
        bias1 = self._bias(token1).squeeze(-1)
        embed2 = self._token_embedding(token2)
        bias2 = self._bias(token2).squeeze(-1)
        dot = (embed1 * embed2).sum(dim=-1)
        output = dot + bias1 + bias2
        return output

    def score_all(self, token):
        """Finds the score of token vs all tokens.

        Args:
          token: Integer index of token to find neighbors of.

        Returns:
          Scores of nearest tokens.
        """
        embed1 = self._token_embedding(token)
        all_tokens = torch.arange(0, self.num_embeddings, dtype=torch.long, device=token.device)
        all_embeds = self._token_embedding(all_tokens)
        scores = torch.matmul(all_embeds, embed1.squeeze(0))
        return scores

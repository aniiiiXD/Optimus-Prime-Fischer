"""Training-only MLP and graph transformer with the same scorer contract."""
import torch
from torch import nn
from features import ADJACENCY, CANDIDATE_DIM


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(CANDIDATE_DIM+64, 64)
        self.mlp2 = nn.Linear(64, 32)
        self.head = nn.Linear(32, 1)

    def forward(self, board, candidate):
        context = torch.cat((board[:, :54].mean(dim=1), board[:, -1]), dim=-1)
        x = torch.cat((candidate, context), dim=-1)
        return self.head(torch.relu(self.mlp2(torch.relu(self.mlp1(x))))).squeeze(-1)


class Transformer(nn.Module):
    def __init__(self, width=24, heads=2):
        super().__init__()
        self.width, self.heads = width, heads
        self.project = nn.Linear(32, width)
        self.position = nn.Parameter(torch.randn(55, width)*.02)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.attn_out = nn.Linear(width, width)
        self.edge_bias = nn.Parameter(torch.full((heads,), .1))
        self.register_buffer("adjacency", torch.as_tensor(ADJACENCY))
        self.norm2 = nn.LayerNorm(width)
        self.ff1, self.ff2 = nn.Linear(width, 2*width), nn.Linear(2*width, width)
        self.norm3 = nn.LayerNorm(width)
        self.candidate = nn.Linear(CANDIDATE_DIM, width)
        self.head1, self.head2 = nn.Linear(3*width, 32), nn.Linear(32, 1)

    def forward(self, board, candidate):
        b, n, _ = board.shape
        x = self.project(board)+self.position
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, self.width//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-2, -1)/(self.width//self.heads)**.5
        scores = scores+self.edge_bias[None, :, None, None]*self.adjacency[None, None]
        joined = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, self.width)
        x = x+self.attn_out(joined)
        x = x+self.ff2(torch.relu(self.ff1(self.norm2(x))))
        tokens = self.norm3(x)
        marked = torch.einsum("bn,bnd->bd", candidate[:, -54:], tokens[:, :54])
        context = torch.cat((tokens[:, -1], marked, torch.relu(self.candidate(candidate))), dim=-1)
        return self.head2(torch.relu(self.head1(context))).squeeze(-1)

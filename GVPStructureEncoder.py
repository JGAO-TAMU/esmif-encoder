import torch
import torch.nn as nn

# GVP-GNN imports (requires torch_geometric & gvp-pytorch)

from torch_geometric.nn import radius_graph
from gvp import GVP, GVPConv

class GVPStructureEncoder(nn.Module):
    """
    Geometric Vector Perceptron Graph Neural Network encoder.
    Builds an E(3)-equivariant GVP message-passing network over residue nodes.

    Inputs:
      - seq_feats: (batch, L, s_dim) scalar node features (e.g., one-hot amino acids)
      - coords: (batch, L, 3) atom coordinates (e.g., C-alpha positions)
    Outputs:
      - embeddings: (batch, L, h_s) per-residue scalar embeddings
    """
    def __init__(
        self,
        seq_dim: int,
        hidden_sv: tuple = (128, 16),
        num_layers: int = 3,
        radius: float = 10.0,
    ):
        super().__init__()
        # radius for graph edges
        self.radius = radius
        # initial GVP projects seq scalars -> hidden scalar & vector dims
        self.input_gvp = GVP((seq_dim, 0), hidden_sv)
        # stack of GVPConv layers (no edge features: edge_dims=(0,0))
        self.convs = nn.ModuleList([
            GVPConv(
                in_dims=hidden_sv,
                out_dims=hidden_sv,
                edge_dims=(0, 0)
            )
            for _ in range(num_layers)
        ])

    def forward(self, seq_feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        seq_feats: (B, L, seq_dim)
        coords:    (B, L, 3)
        returns:   (B, L, hidden_sv[0])
        """
        B, L, _ = coords.size()
        device = coords.device

        # Flatten batch for graph construction
        coords_flat = coords.reshape(B * L, 3)
        seq_flat = seq_feats.reshape(B * L, -1)

        # Build radius‐based graph over coordinates
        edge_index = radius_graph(coords_flat, r=self.radius)

        # No edge features -> create empty scalar & vector edge attrs
        num_edges = edge_index.size(1)
        edge_attr = (
            torch.zeros(num_edges, 0, device=device),      # (E, 0) scalars
            torch.zeros(num_edges, 0, 3, device=device)     # (E, 0, 3) vectors
        )

        # Initial GVP embedding: only scalar features since in_v=0
        h_s, h_v = self.input_gvp(seq_flat)

        # Message‐passing using empty edge_attr
        for conv in self.convs:
            h_s, h_v = conv((h_s, h_v), edge_index, edge_attr)

        # Return just scalar channel, reshaped
        out = h_s.reshape(B, L, -1)
        return out


if __name__ == "__main__":
    batch_size, seq_len = 2, 100
    coords = torch.randn(batch_size, seq_len, 3)
    seq_feats = torch.randn(batch_size, seq_len, 20)  #20 aa one-hot

    # GVP encoder
    gvp_enc = GVPStructureEncoder(seq_dim=20)
    gvp_out = gvp_enc(seq_feats, coords)
    print("GVP-GNN out", gvp_out.shape)

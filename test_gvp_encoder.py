import torch
import matplotlib.pyplot as plt
from GVPStructureEncoder import GVPStructureEncoder
from sklearn.decomposition import PCA

def visualize_embeddings():
    model = GVPStructureEncoder(seq_dim=20)
    coords = torch.randn(1, 200, 3)
    seq_feats = torch.randn(1, 200, 20)
    out = model(seq_feats, coords)[0]  # (L, D)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(out.detach().numpy())

    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("PCA of GVP Encoder Output")
    plt.show()


def test_output_shape():
    for B, L, D in [(1, 50, 20), (2, 100, 20), (4, 75, 30)]:
        coords = torch.randn(B, L, 3)
        seq_feats = torch.randn(B, L, D)
        model = GVPStructureEncoder(seq_dim=D)
        out = model(seq_feats, coords)
        assert out.shape == (B, L, model.input_gvp.out_dims if hasattr(model.input_gvp, 'out_dims') else out.shape[-1])
        print(f"Passed shape test: {out.shape}")

def test_backward_pass():
    B, L, D = 2, 100, 20
    coords = torch.randn(B, L, 3)  # coords used for graph, no grad expected
    seq_feats = torch.randn(B, L, D, requires_grad=True)
    model = GVPStructureEncoder(seq_dim=D)
    
    out = model(seq_feats, coords)
    (out.sum()).backward()
    assert seq_feats.grad is not None, "No gradient w.r.t. seq_feats"
    print("Passed backward-gradient test.")

def test_determinism():
    torch.manual_seed(42)
    B, L, D = 2, 100, 20
    coords = torch.randn(B, L, 3)
    seq_feats = torch.randn(B, L, D)
    model = GVPStructureEncoder(seq_dim=D)
    
    out1 = model(seq_feats, coords)
    out2 = model(seq_feats, coords)
    assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic outputs"
    print("Passed determinism test.")

def test_edge_cases():
    model = GVPStructureEncoder(seq_dim=20)
    
    # All-zero inputs
    coords = torch.zeros(2, 5, 3)
    seq_feats = torch.zeros(2, 5, 20)
    out = model(seq_feats, coords)
    print("Zero-input output shape:", out.shape)

    # Single-residue
    coords = torch.randn(2, 1, 3)
    seq_feats = torch.randn(2, 1, 20)
    out = model(seq_feats, coords)
    print("Single-residue output shape:", out.shape)

def main():
    print("Running GVPStructureEncoder tests...")
    test_output_shape()
    test_backward_pass()
    test_determinism()
    test_edge_cases()
    print("All tests passed.")
    visualize_embeddings()

if __name__ == "__main__":
    main()


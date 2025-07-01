"""python run_gvp_on_pdbs.py <pdb_folder>"""
import os
import sys
import numpy as np
import torch
from Bio.PDB.PDBParser import PDBParser
from GVPStructureEncoder import GVPStructureEncoder

# Mapping 20 standard amino acids to indices
AA_LIST = [
    'ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS',
    'LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL',
    'TRP','TYR'
]
three_to_index = {aa: idx for idx, aa in enumerate(AA_LIST)}

def extract_ca_coords_and_onehot(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    coords = []
    seq_feats = []
    # Process first model, first chain
    model = next(structure.get_models())
    chain = next(model.get_chains())
    for residue in chain:
        if 'CA' not in residue:
            continue
        resname = residue.get_resname()
        if resname not in three_to_index:
            # Skip non-standard residue
            continue
        coords.append(residue['CA'].coord)
        one_hot = np.zeros(len(AA_LIST), dtype=np.float32)
        one_hot[three_to_index[resname]] = 1.0
        seq_feats.append(one_hot)
    coords = np.array(coords, dtype=np.float32)
    seq_feats = np.array(seq_feats, dtype=np.float32)
    return coords, seq_feats

def main(pdb_dir):
    encoder = GVPStructureEncoder(seq_dim=len(AA_LIST))
    encoder.eval()
    for fname in sorted(os.listdir(pdb_dir)):
        if not fname.lower().endswith('.pdb'):
            continue
        path = os.path.join(pdb_dir, fname)
        coords, seq_feats = extract_ca_coords_and_onehot(path)
        if coords.shape[0] == 0:
            print(f"[WARN] No CA residues found in {fname}, skipping.")
            continue
        coords_t = torch.tensor(coords).unsqueeze(0)  # (1, L, 3)
        seq_t = torch.tensor(seq_feats).unsqueeze(0)  # (1, L, 20)
        with torch.no_grad():
            embeddings = encoder(seq_t, coords_t)  # (1, L, hidden_dim)
        embeddings = embeddings.squeeze(0).numpy()  # (L, hidden_dim)
        out_path = os.path.join(pdb_dir, fname.replace('.pdb', '_embeddings.npy'))
        np.save(out_path, embeddings)
        print(f"[OK] {fname} → {embeddings.shape} → {out_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_gvp_on_pdbs.py <pdb_folder>")
        sys.exit(1)
    main(sys.argv[1])

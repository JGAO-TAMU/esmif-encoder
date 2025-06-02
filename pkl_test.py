import torch
import torch.nn as nn
import esm
from esm.inverse_folding.util import extract_coords_from_structure, get_encoder_output
from esm.inverse_folding.util import CoordBatchConverter

from typing import Optional
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure as get_structure_cif
from biotite.structure.io.pdb import PDBFile, get_structure as get_structure_pdb
import biotite.structure as struc  # for filter_peptide_backbone

import pickle

from new_encoder import ESMIFTowerModified

def test_modified_encoder(pkl_file):
    """Test the modified encoder with both PDB and pickle files."""
    
    # Initialize modified encoder
    tower = ESMIFTowerModified(delay_load=False)
    tower = tower.to("cuda")
    # Test with PDB file
    # print("Testing with PDB file...")
    # pdb_features = encoder.forward("sample_protein.pdb", chain="A", file_type="pdb")
    # print(f"PDB features shape: {pdb_features.shape}")
    
    # Test with pickle file
    print(f"Testing with FrameFlow pickle file {pkl_file}...")
    coords = tower.structure_processor(pkl_file)
    pickle_features = tower(coords.to("cuda"))
    print(f"Pickle features shape: {pickle_features.shape}")
    
    # Compare features
    # if pdb_features.shape == pickle_features.shape:
    #     print("Feature shapes are consistent")
    # else:
    #     print("Feature shape mismatch")
    
    return pickle_features

pkl_files_to_test = [
    r"pkl_test_set\00\200l.pkl",
    r"pkl_test_set\0m\10mh.pkl", 
    r"pkl_test_set\01\101m.pkl", 
    r"pkl_test_set\01\201l.pkl", 
    r"pkl_test_set\1b\41bi.pkl", 
    r"pkl_test_set\02\102l.pkl", 
    r"pkl_test_set\2c\12ca.pkl", 
    r"pkl_test_set\04\104l.pkl", 
]

for pkl_file in pkl_files_to_test:
    print(f"######## Testing with {pkl_file} ########")
    test_modified_encoder(pkl_file)
    print(f"######## Finished testing {pkl_file} ########\n")


from original_encoder import ESMIFTower
from new_encoder import ESMIFTowerModified

def test_original_encoder(pdb_path = r"pdb\1a8m.pdb\pdb1a8m.pdb"):
    """Test the current ESMIF encoder with a standard PDB file."""
    
    encoder = ESMIFTower( # Use OriginalESMIFTower
        model_name="esm_if1_gvp4_t16_142M_UR50",
        delay_load=False
    )

    # process structure
    coords_tensor = encoder.structure_processor(pdb_path, chain="A")
    print(f"Coordinates shape: {coords_tensor.shape}")
    print(f"Coordinates dtype: {coords_tensor.dtype}")
    print(f"Sample coordinates:\n{coords_tensor[:3]}")
    
    # generate features
    features = encoder.forward(coords_tensor)
    print(f"Feature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    
    return coords_tensor, features

def test_new_encoder(pdb_path = r"pdb\1a8m.pdb\pdb1a8m.pdb"):
    """ Test the modified ESMIF encoder with a PDB file."""

    encoder = ESMIFTowerModified(
        model_name="esm_if1_gvp4_t16_142M_UR50",
        delay_load=False
    )
    
    # The new encoder's forward method handles structure processing
    features = encoder.forward(file_path=pdb_path, chain="A")
    

    return features # Returning features, coords are processed internally
    
# Execute test
pdb_files_to_test = [
    r"pdb\\1a8m.pdb\\pdb1a8m.pdb",
    #r"pdb\\1bna.pdb\\pdb1bna.pdb", #TypeError
    r"pdb\\1crn.pdb\\pdb1crn.pdb",
    r"pdb\1fat.pdb\pdb1fat.pdb",
    r"pdb\1hho.pdb\pdb1hho.pdb",
    r"pdb\1stp.pdb\pdb1stp.pdb",
]

for pdb_file in pdb_files_to_test:
    print(f"######## Testing with {pdb_file} ########")
    print("-------- Running Original Encoder --------")
    test_original_encoder(pdb_path=pdb_file)
    print("-------- Running New Encoder --------")
    test_new_encoder(pdb_path=pdb_file)
    print(f"######## Finished testing {pdb_file} ########\n")



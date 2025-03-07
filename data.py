import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Union, Callable
import random
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np

class SMILESTokenizer:
    """
    A simple character-level tokenizer for SMILES strings.
    """
    def __init__(self, pad_token="<pad>", unk_token="<unk>", start_token="<s>", end_token="</s>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        
        # Special tokens first
        self.special_tokens = [pad_token, unk_token, start_token, end_token]
        
        # Standard SMILES vocabulary
        self.atom_symbols = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 
                           'S', 'Cl', 'K', 'Ca', 'Br', 'I']
        
        # Add single characters, brackets, and other SMILES syntax characters
        self.syntax_tokens = ['=', '#', '-', '+', '(', ')', '[', ']', '@', '.', ':', '/', '\\']
        
        # Add numbers for ring closures (limited to 0-9 for simplicity)
        self.numbers = [str(i) for i in range(10)]
        
        # Combine all tokens
        self.all_tokens = self.special_tokens + self.atom_symbols + self.syntax_tokens + self.numbers
        
        # Create token to index mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.all_tokens)}
        
        # For atom symbols that are two characters (like 'Cl', 'Br'), we need special handling
        self.multi_char_atoms = set([atom for atom in self.atom_symbols if len(atom) > 1])
        
    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string into a list of tokens.
        
        Args:
            smiles: SMILES string to tokenize
            
        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        
        while i < len(smiles):
            # Check for two-character elements
            if i + 1 < len(smiles) and smiles[i:i+2] in self.multi_char_atoms:
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
                
        return tokens
    
    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a SMILES string into a list of token IDs.
        
        Args:
            smiles: SMILES string to encode
            add_special_tokens: Whether to add start and end tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = [self.start_token] + tokens + [self.end_token]
            
        # Map tokens to indices, using UNK for unknown tokens
        ids = [self.token_to_idx.get(token, self.token_to_idx[self.unk_token]) for token in tokens]
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to a SMILES string.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded SMILES string
        """
        tokens = [self.idx_to_token[idx] for idx in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
            
        return ''.join(tokens)
    
    def __len__(self) -> int:
        """Return the vocabulary size"""
        return len(self.all_tokens)

    
class MolecularDataset(Dataset):
    """
    Dataset for molecular SMILES strings.
    """
    def __init__(
        self, 
        data_path: str, 
        tokenizer: SMILESTokenizer,
        max_length: int = 128,
        filter_fn: Optional[Callable[[str], bool]] = None,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.filter_fn = filter_fn
        
        # Load data from file
        self.data = []
        
        # Support for loading .smi, .csv, and .jsonl files
        file_ext = os.path.splitext(data_path)[1]
        
        if file_ext == '.smi':
            # .smi file format: one SMILES string per line
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and (filter_fn is None or filter_fn(line)):
                        self.data.append(line)
                    if max_samples and len(self.data) >= max_samples:
                        break
                        
        elif file_ext == '.csv':
            # Assumes the CSV has a header and the SMILES column is the first column
            import csv
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row:
                        smiles = row[0].strip()
                        if smiles and (filter_fn is None or filter_fn(smiles)):
                            self.data.append(smiles)
                        if max_samples and len(self.data) >= max_samples:
                            break
                            
        elif file_ext == '.jsonl':
            # Each line is a JSON object with a 'smiles' key
            with open(data_path, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    smiles = obj.get('smiles', '').strip()
                    if smiles and (filter_fn is None or filter_fn(smiles)):
                        self.data.append(smiles)
                    if max_samples and len(self.data) >= max_samples:
                        break
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        print(f"Loaded {len(self.data)} molecules from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        smiles = self.data[idx]
        
        # Tokenize and encode the SMILES string
        token_ids = self.tokenizer.encode(smiles)
        
        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        pad_token_id = self.tokenizer.token_to_idx[self.tokenizer.pad_token]
        padding_length = self.max_length - len(token_ids)
        
        token_ids = token_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'smiles': smiles
        }


# Utility functions for validating molecules and computing rewards
def is_valid_molecule(smiles: str) -> bool:
    """Check if a SMILES string can be parsed by RDKit and sanitized."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def calculate_qed(smiles: str) -> float:
    """Calculate QED score for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return QED.qed(mol)
    except:
        return 0.0
    

def load_chembl_subset(
    output_path: str, 
    num_samples: int = 10000, 
    min_length: int = 5, 
    max_length: int = 100, 
    seed: int = 42
):
    """
    Download a small subset of ChEMBL data and save it to a .smi file.
    
    Args:
        output_path: Path to save the data
        num_samples: Number of molecules to sample
        min_length: Minimum SMILES length
        max_length: Maximum SMILES length
        seed: Random seed for sampling
    """
    try:
        # Try to import the chembl_webresource_client
        from chembl_webresource_client.new_client import new_client
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Get molecules from ChEMBL
        molecule = new_client.molecule
        print("Querying ChEMBL for molecules...")
        
        # Query for small molecules with drug-like properties
        results = molecule.filter(
            molecule_properties__full_mwt__lte=500,
            molecule_properties__alogp__lte=5,
            molecule_properties__full_mwt__gte=200
        ).only(['molecule_chembl_id', 'molecule_structures'])
        
        # Extract SMILES strings
        smiles_list = []
        for res in results:
            if 'molecule_structures' in res and res['molecule_structures']:
                if 'canonical_smiles' in res['molecule_structures']:
                    smiles = res['molecule_structures']['canonical_smiles']
                    if (len(smiles) >= min_length and 
                        len(smiles) <= max_length and 
                        is_valid_molecule(smiles)):
                        smiles_list.append(smiles)
            
            if len(smiles_list) >= num_samples * 2:  # Get more than needed, then sample
                break
                
        # Sample the desired number
        if len(smiles_list) > num_samples:
            smiles_list = random.sample(smiles_list, num_samples)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")
                
        print(f"Saved {len(smiles_list)} molecules to {output_path}")
        
    except ImportError:
        print("chembl_webresource_client not found. Please install with:")
        print("pip install chembl_webresource_client")
        raise


def create_molecular_dataloader(
    data_path: str,
    tokenizer: SMILESTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    filter_fn: Optional[Callable[[str], bool]] = None,
    max_samples: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for molecular data.
    
    Args:
        data_path: Path to the data file
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        filter_fn: Function to filter molecules
        max_samples: Maximum number of samples to load
        
    Returns:
        DataLoader for the dataset
    """
    dataset = MolecularDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        filter_fn=filter_fn,
        max_samples=max_samples
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True
    ) 
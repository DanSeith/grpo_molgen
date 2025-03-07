from typing import List, Dict, Optional, Union, Callable
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski


class MolecularReward:
    """
    Base class for molecular reward functions.
    """
    def __call__(self, smiles: Union[str, List[str]]) -> Union[float, torch.Tensor]:
        """
        Calculate reward for a single SMILES string or a list of SMILES strings.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            
        Returns:
            Reward value(s)
        """
        if isinstance(smiles, str):
            return self.calculate_reward(smiles)
        else:
            rewards = [self.calculate_reward(s) for s in smiles]
            return torch.tensor(rewards, dtype=torch.float)
    
    def calculate_reward(self, smiles: str) -> float:
        """
        Calculate reward for a single SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Reward value
        """
        raise NotImplementedError("Subclasses must implement calculate_reward")


class QEDReward(MolecularReward):
    """
    Reward based on QED (Quantitative Estimate of Drug-likeness).
    Higher QED means the molecule is more drug-like.
    """
    def __init__(self, sanitize: bool = True, validity_penalty: float = 0.0):
        """
        Args:
            sanitize: Whether to sanitize molecules
            validity_penalty: Penalty for invalid molecules (default: 0.0)
        """
        self.sanitize = sanitize
        self.validity_penalty = validity_penalty
    
    def calculate_reward(self, smiles: str) -> float:
        """
        Calculate QED-based reward for a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            QED value (0.0-1.0) or validity_penalty if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.validity_penalty
        
        try:
            if self.sanitize:
                Chem.SanitizeMol(mol)
            return QED.qed(mol)
        except:
            return self.validity_penalty


class ValidityReward(MolecularReward):
    """
    Simple reward based on molecular validity.
    """
    def __init__(self, valid_reward: float = 1.0, invalid_reward: float = 0.0):
        """
        Args:
            valid_reward: Reward for valid molecules
            invalid_reward: Reward for invalid molecules
        """
        self.valid_reward = valid_reward
        self.invalid_reward = invalid_reward
    
    def calculate_reward(self, smiles: str) -> float:
        """
        Check if a SMILES string corresponds to a valid molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            valid_reward if valid, invalid_reward if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.invalid_reward
        
        try:
            Chem.SanitizeMol(mol)
            return self.valid_reward
        except:
            return self.invalid_reward


class CompositeReward(MolecularReward):
    """
    Combine multiple reward functions.
    """
    def __init__(self, reward_functions: List[MolecularReward], weights: Optional[List[float]] = None):
        """
        Args:
            reward_functions: List of reward functions
            weights: List of weights for each reward function (default: equal weights)
        """
        self.reward_functions = reward_functions
        if weights is None:
            self.weights = [1.0] * len(reward_functions)
        else:
            assert len(weights) == len(reward_functions), "Number of weights must match number of reward functions"
            self.weights = weights
    
    def calculate_reward(self, smiles: str) -> float:
        """
        Calculate weighted sum of rewards.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Weighted sum of rewards
        """
        rewards = [func.calculate_reward(smiles) for func in self.reward_functions]
        return sum(r * w for r, w in zip(rewards, self.weights))


class DrugLikenessReward(MolecularReward):
    """
    Reward based on Lipinski's Rule of Five and other drug-likeness properties.
    """
    def __init__(self, validity_penalty: float = 0.0):
        """
        Args:
            validity_penalty: Penalty for invalid molecules
        """
        self.validity_penalty = validity_penalty
    
    def calculate_reward(self, smiles: str) -> float:
        """
        Calculate drug-likeness reward based on multiple properties.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Drug-likeness score (0.0-1.0) or validity_penalty if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.validity_penalty
        
        try:
            # Calculate Lipinski properties
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            
            # Check Lipinski's Rule of Five
            lipinski_pass = (mw <= 500) + (logp <= 5) + (h_donors <= 5) + (h_acceptors <= 10)
            lipinski_score = lipinski_pass / 4.0  # Normalize to 0-1
            
            # QED score
            qed_score = QED.qed(mol)
            
            # Combine scores (equal weight)
            return (lipinski_score + qed_score) / 2.0
        except:
            return self.validity_penalty


class NoveltyReward(MolecularReward):
    """
    Reward based on novelty compared to a reference set.
    """
    def __init__(self, reference_smiles: List[str], validity_penalty: float = 0.0):
        """
        Args:
            reference_smiles: List of reference SMILES strings
            validity_penalty: Penalty for invalid molecules
        """
        self.validity_penalty = validity_penalty
        
        # Create set of canonical SMILES for fast lookup
        self.reference_set = set()
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                self.reference_set.add(canonical_smiles)
    
    def calculate_reward(self, smiles: str) -> float:
        """
        Calculate novelty reward (1.0 if novel, 0.0 if in reference set).
        
        Args:
            smiles: SMILES string
            
        Returns:
            1.0 if novel, 0.0 if in reference set, validity_penalty if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.validity_penalty
        
        try:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return 0.0 if canonical_smiles in self.reference_set else 1.0
        except:
            return self.validity_penalty 
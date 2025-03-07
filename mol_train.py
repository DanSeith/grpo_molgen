import os
import json
import argparse
from pathlib import Path
import random
from typing import Any, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import wandb

from tqdm import tqdm
from rdkit import RDLogger

from models import MolecularLSTM, MolecularGRU, TinyMolecularTransformer
from data import SMILESTokenizer, create_molecular_dataloader, load_chembl_subset
from rewards import CompositeReward, QEDReward, ValidityReward
from loss import GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(model_type: str, vocab_size: int, pad_idx: int) -> nn.Module:
    """
    Initialize a model based on model_type.
    
    Args:
        model_type: Type of model to initialize ('lstm', 'gru', or 'transformer')
        vocab_size: Size of the vocabulary
        pad_idx: Index of the padding token
        
    Returns:
        Initialized model
    """
    if model_type == 'lstm':
        return MolecularLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            pad_idx=pad_idx
        )
    elif model_type == 'gru':
        return MolecularGRU(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            pad_idx=pad_idx
        )
    elif model_type == 'transformer':
        return TinyMolecularTransformer(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            pad_idx=pad_idx
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def molecular_rollout(
    model: nn.Module,
    tokenizer: SMILESTokenizer,
    prefix_token_ids: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    num_rollouts: int,
    reward_fn: Any,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform molecular rollouts.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prefix_token_ids: Prefix token IDs [batch_size, prefix_len]
        prefix_attention_mask: Attention mask for prefix
        num_rollouts: Number of rollouts to perform
        reward_fn: Reward function for evaluating generated molecules
        max_length: Maximum length of generated sequences
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        
    Returns:
        sequence_ids: Generated sequence IDs [num_rollouts, max_length]
        returns: Rewards for each generated sequence [num_rollouts, 1]
        action_mask: Mask for generated tokens [num_rollouts, max_length-1]
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Expand prefix for multiple rollouts
    batch_size = prefix_token_ids.shape[0]
    expanded_prefix_ids = prefix_token_ids.repeat(num_rollouts, 1)
    expanded_prefix_mask = prefix_attention_mask.repeat(num_rollouts, 1)
    
    # Generate completions
    sequence_ids = model.generate(
        input_ids=expanded_prefix_ids,
        attention_mask=expanded_prefix_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.token_to_idx[tokenizer.pad_token],
        eos_token_id=tokenizer.token_to_idx[tokenizer.end_token]
    )
    
    # Create action mask (1 for generated tokens, 0 for prefix and padding)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    for i in range(sequence_ids.shape[0]):
        prefix_len = prefix_attention_mask[i % batch_size].sum().item()
        action_mask[i, prefix_len:] = True
    
    # Set mask to False for padding tokens
    action_mask[sequence_ids == tokenizer.token_to_idx[tokenizer.pad_token]] = False
    
    # Shift action mask for log probs (which are computed for next token prediction)
    action_mask = action_mask[:, :-1]
    
    # Decode sequences and calculate rewards
    generated_smiles = []
    for seq in sequence_ids:
        # Decode the sequence to a SMILES string
        smiles = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        generated_smiles.append(smiles)
    
    # Calculate rewards
    returns = reward_fn(generated_smiles).reshape(-1, 1).to(device)
    
    return sequence_ids, returns, action_mask


def get_sequences_log_probs(
    model: nn.Module,
    sequence_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate log probabilities for token sequences.
    
    Args:
        model: Model to calculate log probs
        sequence_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Log probabilities for each token [batch_size, seq_len-1]
    """
    # Forward pass through the model
    if isinstance(model, (MolecularLSTM, MolecularGRU)):
        logits, _ = model(sequence_ids)
    else:  # TinyMolecularTransformer
        logits = model(sequence_ids, attention_mask)
    
    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs for the actual next tokens
    # We need to shift the targets by 1 to the left
    # logits: [batch_size, seq_len, vocab_size]
    # targets: [batch_size, seq_len-1]
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    
    # Gather the log probs for the actual next tokens
    # result: [batch_size, seq_len-1]
    gathered_log_probs = log_probs[:, :-1].gather(dim=-1, index=targets).squeeze(-1)
    
    return gathered_log_probs


def prepare_batch_for_model(
    model_type: str, 
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare batch for model input.
    
    Args:
        model_type: Type of model ('lstm', 'gru', or 'transformer')
        batch: Batch dictionary with 'input_ids' and 'attention_mask'
        device: Device to put tensors on
        
    Returns:
        input_ids: Input token IDs on device
        attention_mask: Attention mask on device
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description="Train a molecular generator with GRPO")
    parser.add_argument("--model_type", type=str, default="gru", choices=["lstm", "gru", "transformer"],
                        help="Type of model to use")
    parser.add_argument("--data_path", type=str, default="data/chembl_subset.smi",
                        help="Path to training data (SMILES)")
    parser.add_argument("--download_chembl", action="store_true",
                        help="Download ChEMBL subset if data file doesn't exist")
    parser.add_argument("--num_chembl_samples", type=int, default=10000,
                        help="Number of ChEMBL samples to download")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum SMILES length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL divergence weight")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=5, 
                        help="Number of epochs for pretraining")
    parser.add_argument("--rollouts_per_batch", type=int, default=4,
                        help="Number of rollouts per batch item")
    parser.add_argument("--grpo_epochs", type=int, default=3,
                        help="Number of epochs per GRPO step")
    parser.add_argument("--max_norm", type=float, default=1.0, 
                        help="Gradient clipping max norm")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Checkpoint save interval (epochs)")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download ChEMBL subset if needed
    if args.download_chembl and not os.path.exists(args.data_path):
        print(f"Downloading ChEMBL subset to {args.data_path}")
        data_dir = os.path.dirname(args.data_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        load_chembl_subset(
            output_path=args.data_path,
            num_samples=args.num_chembl_samples,
            min_length=5,
            max_length=args.max_length - 10,  # Leave room for special tokens
            seed=args.seed
        )
    
    # Initialize tokenizer
    tokenizer = SMILESTokenizer()
    
    # Create data loader
    train_loader = create_molecular_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        filter_fn=None,
        max_samples=None
    )
    
    # Initialize model
    model = init_model(
        model_type=args.model_type,
        vocab_size=len(tokenizer),
        pad_idx=tokenizer.token_to_idx[tokenizer.pad_token]
    ).to(device)
    
    # Initialize reference model (for KL divergence)
    reference_model = init_model(
        model_type=args.model_type,
        vocab_size=len(tokenizer),
        pad_idx=tokenizer.token_to_idx[tokenizer.pad_token]
    ).to(device)
    reference_model.load_state_dict(model.state_dict())
    reference_model.eval()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize reward function
    reward_fn = CompositeReward([
        QEDReward(validity_penalty=-0.1),  # Reward for drug-likeness
        ValidityReward(valid_reward=0.5, invalid_reward=-0.5)  # Reward for validity
    ], weights=[1.0, 0.5])
    
    # Initialize replay buffer and loss
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=args.clip_eps, kl_weight=args.kl_weight)
    
    # Initialize Weights & Biases
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    else:
        wandb.init(mode="disabled")
    
    # Pretraining phase: train the model to generate valid SMILES
    print("Starting pretraining phase...")
    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        
        for batch in train_loader:
            input_ids, attention_mask = prepare_batch_for_model(
                args.model_type, batch, device
            )
            
            # Forward pass
            if isinstance(model, (MolecularLSTM, MolecularGRU)):
                logits, _ = model(input_ids)
            else:  # TinyMolecularTransformer
                logits = model(input_ids, attention_mask)
            
            # Calculate loss (next token prediction)
            # Shift targets by 1 position
            targets = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            
            # Create mask to ignore padding in loss
            mask = attention_mask[:, 1:].contiguous()
            
            # Calculate cross entropy loss with masking
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.token_to_idx[tokenizer.pad_token],
                reduction='none'
            ).view_as(targets) * mask
            
            # Average loss over non-padding tokens
            loss = loss.sum() / mask.sum()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        # Update reference model at the end of each epoch
        reference_model.load_state_dict(model.state_dict())
        
        # Log and save
        avg_loss = total_loss / batches
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"pretrain/loss": avg_loss, "epoch": epoch})
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, output_dir / f"pretrain_checkpoint_{epoch+1}.pt")
    
    # GRPO training phase
    print("Starting GRPO training phase...")
    for epoch in tqdm(range(args.num_epochs), desc="GRPO epochs"):
        # Clear replay buffer at the start of each epoch
        replay_buffer.clear()
        
        # 1. Collect trajectories with the current policy
        model.eval()
        total_reward = 0.0
        total_rollouts = 0
        
        rollout_bar = tqdm(train_loader, desc=f"Rollouts for epoch {epoch+1}/{args.num_epochs}", leave=False)
        for batch in rollout_bar:
            input_ids, attention_mask = prepare_batch_for_model(
                args.model_type, batch, device
            )
            
            # Generate molecules using rollouts
            sequence_ids, returns, action_mask = molecular_rollout(
                model=model,
                tokenizer=tokenizer,
                prefix_token_ids=input_ids,
                prefix_attention_mask=attention_mask,
                num_rollouts=args.rollouts_per_batch,
                reward_fn=reward_fn,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Calculate log probabilities for the current and reference models
            log_probs = get_sequences_log_probs(model, sequence_ids, attention_mask=None)
            with torch.no_grad():
                log_probs_ref = get_sequences_log_probs(reference_model, sequence_ids, attention_mask=None)
            
            # Calculate advantages (normalize returns)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Create and store experience
            experience = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                returns=returns,
                advantages=advantages,
                attention_mask=None,  # We use action_mask instead
                action_mask=action_mask
            )
            
            replay_buffer.append(experience)
            
            total_reward += returns.sum().item()
            total_rollouts += sequence_ids.shape[0]
            
            current_avg_reward = total_reward / total_rollouts if total_rollouts > 0 else 0
            rollout_bar.set_postfix(avg_reward=f"{current_avg_reward:.4f}")
            
            # Print sample generations at each epoch
            if len(replay_buffer) > 0 and total_rollouts <= args.rollouts_per_batch:
                for i in range(min(3, sequence_ids.shape[0])):
                    smiles = tokenizer.decode(sequence_ids[i].tolist(), skip_special_tokens=True)
                    reward = returns[i].item()
                    print(f"Generated: {smiles}, Reward: {reward:.4f}")
        
        # 2. Train policy with GRPO for multiple epochs
        policy_bar = tqdm(range(args.grpo_epochs), desc="GRPO policy update", leave=False)
        for _ in policy_bar:
            model.train()
            
            # Process all experiences in the replay buffer
            experiences = join_experience_batch(replay_buffer.items)
            
            # Get sequence log probs with current policy
            log_probs = get_sequences_log_probs(
                model, 
                experiences.sequences, 
                attention_mask=None
            )
            
            # Calculate loss
            loss, kl = objective(log_probs, experiences)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            policy_bar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl.item():.4f}")
        
        # 3. Update reference model at the end of each epoch
        reference_model.load_state_dict(model.state_dict())
        
        # Log and save
        avg_reward = total_reward / total_rollouts if total_rollouts > 0 else 0
        print(f"Epoch {epoch+1}/{args.num_epochs}, Avg Reward: {avg_reward:.4f}, Loss: {loss.item():.4f}, KL: {kl.item():.4f}")
        wandb.log({
            "train/avg_reward": avg_reward,
            "train/loss": loss.item(),
            "train/kl": kl.item(),
            "epoch": epoch + args.pretrain_epochs
        })
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch + args.pretrain_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'reward': avg_reward,
            }, output_dir / f"grpo_checkpoint_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs + args.pretrain_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / "final_model.pt")
    
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main() 
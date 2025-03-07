import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularLSTM(nn.Module):
    """
    A simple LSTM-based model for generating SMILES strings.
    Much lighter than a full LLM, but still capable of learning molecular patterns.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
            hidden: Hidden state tuple (h_0, c_0) for the LSTM
            
        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
            hidden: Final hidden state tuple
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # LSTM forward pass
        # output: [batch_size, seq_len, hidden_dim]
        # hidden: (h_n, c_n) where h_n/c_n is [num_layers, batch_size, hidden_dim]
        output, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout and project to vocabulary size
        output = self.dropout(output)
        logits = self.fc_out(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=None
    ):
        """
        Generate sequences autoregressively.
        
        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            attention_mask: Mask for padding [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or take argmax
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            generated_ids: Generated token IDs [batch_size, max_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        seq_len = input_ids.shape[1]
        
        # Create output tensor filled with padding
        generated_ids = torch.full(
            (batch_size, max_length), 
            pad_token_id,
            dtype=torch.long, 
            device=device
        )
        
        # Copy input_ids to the beginning
        generated_ids[:, :seq_len] = input_ids
        
        # Track which sequences are still being generated
        unfinished_seqs = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Start with a clean hidden state
        hidden = self.init_hidden(batch_size, device)
        
        # Encode the prefix
        _, hidden = self.forward(input_ids, hidden)
        
        # Generate autoregressively
        cur_len = seq_len
        while cur_len < max_length and unfinished_seqs.any():
            # Get the last token for each sequence
            current_token = generated_ids[:, cur_len-1:cur_len]
            
            # Forward pass
            logits, hidden = self.forward(current_token, hidden)
            
            # Get the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling
            if do_sample and top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add token to output
            generated_ids[:, cur_len] = next_tokens
            
            # Check if we've generated EOS tokens
            if eos_token_id is not None:
                unfinished_seqs = unfinished_seqs & (next_tokens != eos_token_id)
            
            cur_len += 1
        
        return generated_ids
    
    
class MolecularGRU(nn.Module):
    """
    A simple GRU-based model for generating SMILES strings.
    Lighter and often faster to train than LSTM.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
            hidden: Hidden state for the GRU
            
        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
            hidden: Final hidden state
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # GRU forward pass
        # output: [batch_size, seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        output, hidden = self.gru(embedded, hidden)
        
        # Apply dropout and project to vocabulary size
        output = self.dropout(output)
        logits = self.fc_out(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=None
    ):
        """
        Generate sequences autoregressively.
        
        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            attention_mask: Mask for padding [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or take argmax
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            generated_ids: Generated token IDs [batch_size, max_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        seq_len = input_ids.shape[1]
        
        # Create output tensor filled with padding
        generated_ids = torch.full(
            (batch_size, max_length), 
            pad_token_id,
            dtype=torch.long, 
            device=device
        )
        
        # Copy input_ids to the beginning
        generated_ids[:, :seq_len] = input_ids
        
        # Track which sequences are still being generated
        unfinished_seqs = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Start with a clean hidden state
        hidden = self.init_hidden(batch_size, device)
        
        # Encode the prefix
        _, hidden = self.forward(input_ids, hidden)
        
        # Generate autoregressively
        cur_len = seq_len
        while cur_len < max_length and unfinished_seqs.any():
            # Get the last token for each sequence
            current_token = generated_ids[:, cur_len-1:cur_len]
            
            # Forward pass
            logits, hidden = self.forward(current_token, hidden)
            
            # Get the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling
            if do_sample and top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add token to output
            generated_ids[:, cur_len] = next_tokens
            
            # Check if we've generated EOS tokens
            if eos_token_id is not None:
                unfinished_seqs = unfinished_seqs & (next_tokens != eos_token_id)
            
            cur_len += 1
        
        return generated_ids
    

class TinyMolecularTransformer(nn.Module):
    """
    A tiny transformer model for molecular generation.
    Much smaller than standard LLMs but more powerful than LSTM/GRU models.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x, attention_mask=None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
            attention_mask: Attention mask for padding
            
        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        
        # Create padding mask if attention_mask is provided
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        
        # Transformer forward pass
        # output: [batch_size, seq_len, embedding_dim]
        output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Project to vocabulary size
        logits = self.fc_out(output)
        
        return logits
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=None
    ):
        """
        Generate sequences autoregressively.
        
        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            attention_mask: Mask for padding [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or take argmax
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            generated_ids: Generated token IDs [batch_size, max_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        seq_len = input_ids.shape[1]
        
        # Create output tensor filled with padding
        generated_ids = torch.full(
            (batch_size, max_length), 
            pad_token_id,
            dtype=torch.long, 
            device=device
        )
        
        # Copy input_ids to the beginning
        generated_ids[:, :seq_len] = input_ids
        
        # Track which sequences are still being generated
        unfinished_seqs = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Generate autoregressively
        cur_len = seq_len
        while cur_len < max_length and unfinished_seqs.any():
            # Create attention mask for current sequence
            cur_attention_mask = None
            if attention_mask is not None:
                # Extend attention mask to current length
                cur_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, cur_len - seq_len, device=device)
                ], dim=1)
            
            # Forward pass
            logits = self.forward(
                generated_ids[:, :cur_len],
                attention_mask=cur_attention_mask
            )
            
            # Get the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling
            if do_sample and top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add token to output
            generated_ids[:, cur_len] = next_tokens
            
            # Check if we've generated EOS tokens
            if eos_token_id is not None:
                unfinished_seqs = unfinished_seqs & (next_tokens != eos_token_id)
            
            cur_len += 1
        
        return generated_ids


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1), 0].unsqueeze(0)
        return self.dropout(x) 
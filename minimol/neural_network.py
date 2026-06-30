"""
Core Neural Network Module for Minimol 70B Parameter Transformer
Implements transformer architecture with RoPE positional embeddings,
multi-head attention, and complete training pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Tuple, Dict, List
import os
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    Applies rotation to query and key vectors based on position
    Reference: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_seq_length: int = 32000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Pre-compute freqs for efficiency
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            seq_len: Sequence length (if None, uses x.shape[1])
        
        Returns:
            Tensor with rotary embeddings applied
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Compute position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies for each position
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Expand freqs to match dimensions
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Apply rotation
        cos_cached = emb.cos()[None, :, :]
        sin_cached = emb.sin()[None, :, :]
        
        # Rotate embeddings
        x_rot = (x * cos_cached) + self._rotate_half(x) * sin_cached
        
        return x_rot
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with 64 heads
    Uses RoPE for positional encoding
    """
    
    def __init__(self, hidden_dim: int = 4096, num_heads: int = 64, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for multi-head attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            past_kv: Cached key-value tensors from previous steps
        
        Returns:
            Tuple of (output, (cached_k, cached_v))
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project input to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # Use cached K, V if available (for inference optimization)
        if past_kv is not None:
            cached_k, cached_v = past_kv
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, (k, v)


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN)
    Two-layer network with ReLU activation
    """
    
    def __init__(self, hidden_dim: int = 4096, ffn_dim: int = 11008, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    Combines multi-head attention and feed-forward networks with residual connections
    """
    
    def __init__(self, hidden_dim: int = 4096, num_heads: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, hidden_dim * 2.75, dropout)  # ~11B params per layer
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor
            attention_mask: Attention mask
            past_kv: Cached key-value from attention
        
        Returns:
            Tuple of (output, cached_kv)
        """
        # Attention with residual connection
        norm_x = self.norm1(x)
        attn_out, kv_cache = self.attention(norm_x, attention_mask, past_kv)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        
        return x, kv_cache


class Minimol70B(nn.Module):
    """
    Minimol 70B Parameter Transformer Neural Network
    
    Architecture:
    - 80 transformer blocks (64 heads × 4096 hidden dim)
    - ~70B total parameters
    - 32K token context window
    - RoPE positional embeddings
    - LayerNorm, gradient clipping, AdamW optimizer
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 4096,
        num_layers: int = 80,
        num_heads: int = 64,
        max_seq_length: int = 32000,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.device_name = device
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(device)
        
        # Calculate approximate parameter count
        self.param_count = self._count_parameters()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _count_parameters(self) -> int:
        """Count total parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask (1 for attend, -inf for mask)
            past_kv_list: Cached key-value tensors from previous steps
        
        Returns:
            Tuple of (logits, kv_cache_list)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device, dtype=input_ids.dtype)
        
        # Get embeddings
        x = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = x + pos_emb
        x = self.dropout(x)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Pass through transformer layers
        new_kv_list = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, kv_cache = layer(x, attention_mask, past_kv)
            new_kv_list.append(kv_cache)
        
        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits, new_kv_list
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = (1.0 - mask) * -10000.0
        return mask
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text using the model
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            Generated token IDs
        """
        self.eval()
        device = next(self.parameters()).device
        
        generated = input_ids.clone()
        past_kv_list = None
        
        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="Generating"):
                # Get last token
                last_token_id = generated[:, -1:]
                
                # Forward pass
                logits, past_kv_list = self.forward(last_token_id, past_kv_list=past_kv_list)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k and top-p sampling
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


class MinimolTrainer:
    """
    Training pipeline for Minimol 70B model
    Handles forward/backward passes, checkpointing, and validation
    """
    
    def __init__(
        self,
        model: Minimol70B,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer with AdamW
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-5,
        )
        
        self.training_history = {
            "loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
        
        self.global_step = 0
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        Single training step
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs for loss computation
        
        Returns:
            Loss value
        """
        self.model.train()
        
        # Forward pass
        logits, _ = self.model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.model.vocab_size),
            target_ids.view(-1),
            ignore_index=-100,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def eval_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        Validation step
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
        
        Returns:
            Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, self.model.vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
            )
        
        return loss.item()
    
    def train_epoch(
        self,
        train_dataloader,
        val_dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epoch: Epoch number
        
        Returns:
            Dictionary with training metrics
        """
        train_losses = []
        
        # Training
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            loss = self.train_step(input_ids, target_ids)
            train_losses.append(loss)
            
            progress_bar.set_postfix({"loss": loss})
        
        # Validation
        val_losses = []
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]")
        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            loss = self.eval_step(input_ids, target_ids)
            val_losses.append(loss)
            
            progress_bar.set_postfix({"loss": loss})
        
        # Compute metrics
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        lr = self.optimizer.param_groups[0]["lr"]
        
        # Store in history
        self.training_history["loss"].append(avg_train_loss)
        self.training_history["val_loss"].append(avg_val_loss)
        self.training_history["learning_rate"].append(lr)
        
        return {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": lr,
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """
        Save model checkpoint
        
        Args:
            epoch: Epoch number
            metrics: Training metrics
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
            "training_history": self.training_history,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.training_history = checkpoint.get("training_history", self.training_history)
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")
        
        return checkpoint["epoch"]
    
    def save_training_history(self, filepath: str = "training_history.json"):
        """Save training history to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Minimol 70B Parameter Transformer - Core Neural Network")
    print("=" * 80)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    model = Minimol70B(
        vocab_size=50257,
        hidden_dim=4096,
        num_layers=80,
        num_heads=64,
        max_seq_length=32000,
        dropout=0.1,
        device=device,
    )
    
    print(f"\nModel initialized successfully!")
    print(f"Total parameters: {model.param_count:,.0f}")
    print(f"Approximate size: {model.param_count * 4 / (1024**3):.2f} GB (fp32)")
    
    # Test forward pass
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output vocabulary dimension: {logits.shape[-1]}")
    
    # Test generation
    print("\n" + "=" * 80)
    print("Testing text generation...")
    print("=" * 80)
    
    prompt = torch.randint(0, 50257, (1, 10)).to(device)
    generated = model.generate(prompt, max_length=20, temperature=0.7)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Sample generated tokens: {generated[0][:30].tolist()}")

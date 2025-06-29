#!/usr/bin/env python3
"""
Data Science Interview Prep - Generative AI & Foundation Models
==============================================================
Implementations of attention mechanisms, transformers, and LLM concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import math
import time
from dataclasses import dataclass

# Try importing PyTorch and Transformers
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not installed. Some examples will use NumPy only.")
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from datasets import Dataset as HFDataset
    import evaluate

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers library not installed. Some examples will be skipped.")
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# ATTENTION MECHANISM FROM SCRATCH
# ============================================================================

class AttentionMechanisms:
    """Various attention mechanisms implemented from scratch."""

    @staticmethod
    def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray,
                                     value: np.ndarray, mask: Optional[np.ndarray] = None,
                                     visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

        Args:
            query: Query matrix [batch_size, seq_len, d_k]
            key: Key matrix [batch_size, seq_len, d_k]
            value: Value matrix [batch_size, seq_len, d_v]
            mask: Optional mask [batch_size, seq_len, seq_len]
            visualize: Whether to visualize attention weights

        Returns:
            output: Attention output [batch_size, seq_len, d_v]
            attention_weights: Attention weights [batch_size, seq_len, seq_len]
        """
        # Get dimensions
        d_k = query.shape[-1]

        # Calculate attention scores
        scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = AttentionMechanisms._softmax(scores, axis=-1)

        # Apply attention to values
        output = np.matmul(attention_weights, value)

        if visualize and len(query.shape) == 3:
            AttentionMechanisms._visualize_attention(attention_weights[0])

        return output, attention_weights

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax implementation."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def _visualize_attention(attention_weights: np.ndarray, tokens: Optional[List[str]] = None):
        """Visualize attention weights as heatmap."""
        plt.figure(figsize=(10, 8))

        if tokens is None:
            tokens = [f'Token_{i}' for i in range(attention_weights.shape[0])]

        sns.heatmap(attention_weights,
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap='Blues',
                    cbar_kws={'label': 'Attention Weight'})

        plt.title('Attention Weights Visualization')
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def multi_head_attention_numpy(query: np.ndarray, key: np.ndarray, value: np.ndarray,
                                   n_heads: int = 8, d_model: int = 512) -> np.ndarray:
        """
        Multi-head attention implementation in NumPy.

        Shows the core concept without PyTorch dependencies.
        """
        batch_size, seq_len, _ = query.shape
        d_k = d_model // n_heads

        # Initialize weight matrices (would be learned in practice)
        W_q = np.random.randn(n_heads, d_model, d_k) * 0.01
        W_k = np.random.randn(n_heads, d_model, d_k) * 0.01
        W_v = np.random.randn(n_heads, d_model, d_k) * 0.01
        W_o = np.random.randn(d_model, d_model) * 0.01

        # Linear transformations and split into heads
        Q = np.stack([query @ W_q[i] for i in range(n_heads)], axis=1)
        K = np.stack([key @ W_k[i] for i in range(n_heads)], axis=1)
        V = np.stack([value @ W_v[i] for i in range(n_heads)], axis=1)

        # Apply attention for each head
        attention_outputs = []
        for i in range(n_heads):
            output, _ = AttentionMechanisms.scaled_dot_product_attention(
                Q[:, i], K[:, i], V[:, i]
            )
            attention_outputs.append(output)

        # Concatenate heads
        concat_attention = np.concatenate(attention_outputs, axis=-1)

        # Final linear transformation
        output = concat_attention @ W_o

        return output


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding:
    """Positional encoding for transformer models."""

    @staticmethod
    def sinusoidal_encoding(seq_len: int, d_model: int,
                            visualize: bool = False) -> np.ndarray:
        """
        Create sinusoidal positional encodings.

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            seq_len: Sequence length
            d_model: Model dimension
            visualize: Whether to visualize the encodings

        Returns:
            Positional encoding matrix [seq_len, d_model]
        """
        pe = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len).reshape(-1, 1)

        # Create div_term for even and odd dimensions
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)

        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])

        if visualize:
            PositionalEncoding._visualize_encoding(pe)

        return pe

    @staticmethod
    def _visualize_encoding(pe: np.ndarray):
        """Visualize positional encoding patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Heatmap of positional encodings
        im = ax1.imshow(pe.T, aspect='auto', cmap='RdBu_r')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Dimension')
        ax1.set_title('Positional Encoding Heatmap')
        plt.colorbar(im, ax=ax1)

        # Plot some specific dimensions
        positions = range(pe.shape[0])
        for dim in [0, 1, 10, 11, 100, 101]:
            if dim < pe.shape[1]:
                ax2.plot(positions, pe[:, dim], label=f'Dim {dim}')

        ax2.set_xlabel('Position')
        ax2.set_ylabel('Encoding Value')
        ax2.set_title('Positional Encoding for Selected Dimensions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def learned_encoding(seq_len: int, d_model: int) -> np.ndarray:
        """
        Learned positional embeddings (initialized randomly).
        In practice, these would be learned during training.
        """
        return np.random.randn(seq_len, d_model) * 0.01


# ============================================================================
# TRANSFORMER COMPONENTS (PYTORCH)
# ============================================================================

if TORCH_AVAILABLE:

    class MultiHeadAttention(nn.Module):
        """
        Multi-head attention mechanism.

        Key concepts for interviews:
        - Parallel attention heads
        - Linear projections for Q, K, V
        - Scaled dot-product attention
        - Concatenation and final projection
        """

        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            assert d_model % n_heads == 0

            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            # Linear projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)
            seq_len = query.size(1)

            # Linear transformations and reshape for multiple heads
            Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # Attention
            attention_output, attention_weights = self.scaled_dot_product_attention(
                Q, K, V, mask
            )

            # Concatenate heads
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )

            # Final linear transformation
            output = self.W_o(attention_output)

            return output, attention_weights

        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            output = torch.matmul(attention_weights, V)

            return output, attention_weights


    class PositionwiseFeedForward(nn.Module):
        """
        Position-wise feed-forward network.

        FFN(x) = max(0, xW1 + b1)W2 + b2
        """

        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.dropout(self.activation(self.fc1(x))))


    class TransformerBlock(nn.Module):
        """
        Single transformer encoder block.

        Components:
        - Multi-head self-attention
        - Add & Norm
        - Feed-forward network
        - Add & Norm
        """

        def __init__(self, d_model: int, n_heads: int, d_ff: int,
                     dropout: float = 0.1):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Self-attention with residual connection
            attn_output, _ = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-forward with residual connection
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))

            return x


    class ToyGPT(nn.Module):
        """
        Simplified GPT model for educational purposes.

        Architecture:
        - Token embeddings
        - Positional encodings
        - Stack of transformer blocks
        - Output projection
        """

        def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                     n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512,
                     dropout: float = 0.1):
            super().__init__()

            self.d_model = d_model
            self.max_seq_len = max_seq_len

            # Embeddings
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

            # Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

            self.layer_norm = nn.LayerNorm(d_model)
            self.output_projection = nn.Linear(d_model, vocab_size)

            self.dropout = nn.Dropout(dropout)

            # Initialize weights
            self.init_weights()

        def init_weights(self):
            """Initialize weights with Xavier uniform."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape

            # Token embeddings
            token_embeddings = self.token_embedding(input_ids)

            # Positional embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            position_embeddings = self.position_embedding(positions)

            # Combine embeddings
            embeddings = self.dropout(token_embeddings + position_embeddings)

            # Pass through transformer blocks
            hidden_states = embeddings
            for transformer_block in self.transformer_blocks:
                hidden_states = transformer_block(hidden_states, attention_mask)

            # Final layer norm and projection
            hidden_states = self.layer_norm(hidden_states)
            logits = self.output_projection(hidden_states)

            return logits

        @torch.no_grad()
        def generate(self, input_ids, max_new_tokens: int = 50,
                     temperature: float = 1.0, top_k: int = 50):
            """
            Generate text autoregressively.

            Args:
                input_ids: Starting token IDs [batch_size, seq_len]
                max_new_tokens: Number of tokens to generate
                temperature: Sampling temperature (higher = more random)
                top_k: Only sample from top k tokens

            Returns:
                Generated token IDs
            """
            self.eval()

            for _ in range(max_new_tokens):
                # Crop input if it exceeds max sequence length
                input_ids_crop = input_ids[:, -self.max_seq_len:]

                # Get predictions
                logits = self(input_ids_crop)
                logits = logits[:, -1, :]  # Focus on last position

                # Apply temperature
                logits = logits / temperature

                # Top-k sampling
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, indices, values)

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if EOS token is generated (assuming EOS token ID is 2)
                if (next_token == 2).any():
                    break

            return input_ids


# ============================================================================
# TOKENIZATION
# ============================================================================

class SimpleTokenizer:
    """
    Simple character-level tokenizer for demonstration.

    In practice, you'd use BPE, WordPiece, or SentencePiece.
    """

    def __init__(self, text_corpus: str):
        # Get unique characters
        self.chars = sorted(list(set(text_corpus)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Add special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'

        special_tokens = [self.pad_token, self.unk_token,
                          self.bos_token, self.eos_token]

        for token in special_tokens:
            if token not in self.char_to_idx:
                self.char_to_idx[token] = self.vocab_size
                self.idx_to_char[self.vocab_size] = token
                self.vocab_size += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_idx[self.bos_token])

        for ch in text:
            if ch in self.char_to_idx:
                tokens.append(self.char_to_idx[ch])
            else:
                tokens.append(self.char_to_idx[self.unk_token])

        if add_special_tokens:
            tokens.append(self.char_to_idx[self.eos_token])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        chars = []
        special_ids = {self.char_to_idx[token] for token in
                       [self.pad_token, self.unk_token, self.bos_token, self.eos_token]}

        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])

        return ''.join(chars)

    def batch_encode(self, texts: List[str], max_length: int = None,
                     padding: bool = True) -> Dict[str, np.ndarray]:
        """Encode multiple texts with padding."""
        encoded = [self.encode(text) for text in texts]

        if max_length is None:
            max_length = max(len(seq) for seq in encoded)

        # Truncate or pad sequences
        input_ids = []
        attention_mask = []

        for seq in encoded:
            if len(seq) > max_length:
                seq = seq[:max_length]

            mask = [1] * len(seq)

            if padding and len(seq) < max_length:
                padding_length = max_length - len(seq)
                seq.extend([self.char_to_idx[self.pad_token]] * padding_length)
                mask.extend([0] * padding_length)

            input_ids.append(seq)
            attention_mask.append(mask)

        return {
            'input_ids': np.array(input_ids),
            'attention_mask': np.array(attention_mask)
        }


def demonstrate_tokenization():
    """Demonstrate different tokenization approaches."""
    print("=== Tokenization Demo ===")

    sample_text = "Hello, world! This is a tokenization example."

    # Character-level tokenization
    char_tokenizer = SimpleTokenizer(sample_text)
    encoded = char_tokenizer.encode(sample_text)
    decoded = char_tokenizer.decode(encoded)

    print(f"Original text: {sample_text}")
    print(f"Vocabulary size: {char_tokenizer.vocab_size}")
    print(f"Encoded (first 20 tokens): {encoded[:20]}...")
    print(f"Decoded: {decoded}")

    # Demonstrate batch encoding
    texts = [
        "Short text.",
        "This is a longer text that will need padding.",
        "Medium length text here."
    ]

    batch_encoded = char_tokenizer.batch_encode(texts, max_length=30)
    print(f"\nBatch encoding shape: {batch_encoded['input_ids'].shape}")
    print(f"Attention mask shape: {batch_encoded['attention_mask'].shape}")


# ============================================================================
# PROMPT ENGINEERING
# ============================================================================

class PromptEngineering:
    """Examples of different prompting techniques."""

    @staticmethod
    def zero_shot_prompt(task: str, input_text: str) -> str:
        """Zero-shot prompting - no examples given."""
        return f"""Task: {task}

Input: {input_text}

Output:"""

    @staticmethod
    def few_shot_prompt(task: str, examples: List[Tuple[str, str]],
                        input_text: str) -> str:
        """Few-shot prompting - provide examples."""
        prompt = f"Task: {task}\n\n"

        for i, (example_input, example_output) in enumerate(examples):
            prompt += f"Example {i + 1}:\n"
            prompt += f"Input: {example_input}\n"
            prompt += f"Output: {example_output}\n\n"

        prompt += f"Now, complete this:\n"
        prompt += f"Input: {input_text}\n"
        prompt += f"Output:"

        return prompt

    @staticmethod
    def chain_of_thought_prompt(problem: str) -> str:
        """Chain-of-thought prompting - encourage step-by-step reasoning."""
        return f"""Problem: {problem}

Let's solve this step by step:
1. First, let's understand what we're asked to find.
2. Next, let's identify the relevant information.
3. Then, let's work through the solution.
4. Finally, let's verify our answer.

Solution:"""

    @staticmethod
    def instruction_following_prompt(instruction: str, context: str = None) -> str:
        """Instruction-following prompt format."""
        prompt = f"### Instruction:\n{instruction}\n\n"

        if context:
            prompt += f"### Context:\n{context}\n\n"

        prompt += "### Response:"

        return prompt

    @staticmethod
    def structured_output_prompt(task: str, output_format: str, input_text: str) -> str:
        """Prompt for structured output (JSON, XML, etc.)."""
        return f"""Task: {task}

Required Output Format:
{output_format}

Input: {input_text}

Structured Output:"""


def demonstrate_prompting_techniques():
    """Show different prompting techniques with examples."""
    print("=== Prompt Engineering Examples ===\n")

    # 1. Zero-shot
    print("1. Zero-shot Prompting:")
    print("-" * 40)
    zero_shot = PromptEngineering.zero_shot_prompt(
        task="Classify the sentiment as positive, negative, or neutral",
        input_text="This movie was absolutely fantastic!"
    )
    print(zero_shot)

    # 2. Few-shot
    print("\n\n2. Few-shot Prompting:")
    print("-" * 40)
    examples = [
        ("I love this product!", "positive"),
        ("This is terrible.", "negative"),
        ("It's okay, nothing special.", "neutral")
    ]
    few_shot = PromptEngineering.few_shot_prompt(
        task="Classify the sentiment",
        examples=examples,
        input_text="Best purchase I've ever made!"
    )
    print(few_shot)

    # 3. Chain-of-thought
    print("\n\n3. Chain-of-Thought Prompting:")
    print("-" * 40)
    cot = PromptEngineering.chain_of_thought_prompt(
        problem="If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?"
    )
    print(cot)

    # 4. Structured output
    print("\n\n4. Structured Output Prompting:")
    print("-" * 40)
    structured = PromptEngineering.structured_output_prompt(
        task="Extract person information",
        output_format="""{
    "name": "string",
    "age": "number",
    "occupation": "string",
    "location": "string"
}""",
        input_text="John Smith, 28 years old, works as a software engineer in San Francisco."
    )
    print(structured)


# ============================================================================
# FINE-TUNING EXAMPLE
# ============================================================================

def demonstrate_fine_tuning():
    """Demonstrate fine-tuning setup with HuggingFace."""
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers library not available. Skipping fine-tuning demo.")
        return

    print("\n=== Fine-Tuning Setup Demo ===")

    # Example: Fine-tuning for sentiment classification
    model_name = "distilbert-base-uncased"

    # Sample data
    texts = [
        "I love this movie!",
        "This is terrible.",
        "Not bad, quite good actually.",
        "Worst experience ever."
    ]
    labels = [1, 0, 1, 0]  # 1: positive, 0: negative

    print(f"Model: {model_name}")
    print(f"Task: Binary sentiment classification")
    print(f"Training samples: {len(texts)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    print("\nTraining Configuration:")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Warmup steps: {training_args.warmup_steps}")
    print(f"  - Weight decay: {training_args.weight_decay}")

    # LoRA configuration for parameter-efficient fine-tuning
    print("\n=== LoRA Configuration Example ===")
    lora_config = {
        "r": 16,  # Rank
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],  # Target attention modules
        "bias": "none"
    }

    print("LoRA Parameters:")
    for key, value in lora_config.items():
        print(f"  - {key}: {value}")

    # Calculate parameter reduction
    original_params = 66_362_880  # DistilBERT parameters
    lora_params = 2 * 768 * 16 * 2  # Approximate LoRA parameters
    reduction = (1 - lora_params / original_params) * 100

    print(f"\nParameter Reduction with LoRA: {reduction:.1f}%")
    print(f"Trainable parameters: {lora_params:,} (vs {original_params:,} original)")


# ============================================================================
# GENERATION EVALUATION METRICS
# ============================================================================

class GenerationMetrics:
    """Metrics for evaluating text generation quality."""

    @staticmethod
    def calculate_perplexity(log_probs: np.ndarray) -> float:
        """
        Calculate perplexity from log probabilities.
        Perplexity = exp(-1/N * sum(log P(w_i)))
        """
        avg_log_prob = np.mean(log_probs)
        return np.exp(-avg_log_prob)

    @staticmethod
    def calculate_bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
        """
        Simplified BLEU score calculation.
        In practice, use nltk.translate.bleu_score or sacrebleu.
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()

        # Calculate n-gram precision
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = [tuple(ref_tokens[j:j + i]) for j in range(len(ref_tokens) - i + 1)]
            hyp_ngrams = [tuple(hyp_tokens[j:j + i]) for j in range(len(hyp_tokens) - i + 1)]

            if not hyp_ngrams:
                precisions.append(0)
                continue

            matches = sum(1 for ngram in hyp_ngrams if ngram in ref_ngrams)
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)

        # Brevity penalty
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(hyp_tokens))

        # BLEU score (geometric mean of precisions)
        if all(p > 0 for p in precisions):
            bleu = bp * np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0

        return bleu

    @staticmethod
    def calculate_diversity(texts: List[str], n: int = 3) -> Dict[str, float]:
        """
        Calculate diversity metrics for generated texts.

        Returns:
            - distinct-n: Ratio of unique n-grams
            - entropy: Entropy of n-gram distribution
        """
        all_ngrams = []

        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return {"distinct_n": 0, "entropy": 0}

        # Distinct-n
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        distinct_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

        # Entropy
        ngram_counts = {}
        for ngram in all_ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        probs = np.array(list(ngram_counts.values())) / total_ngrams
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return {
            f"distinct_{n}": distinct_n,
            f"entropy_{n}": entropy
        }

    @staticmethod
    def human_eval_template() -> Dict[str, List[str]]:
        """Template for human evaluation criteria."""
        return {
            "fluency": [
                "Is the text grammatically correct?",
                "Does it read naturally?",
                "Are there any awkward phrasings?"
            ],
            "coherence": [
                "Does the text flow logically?",
                "Are ideas connected well?",
                "Is there a clear structure?"
            ],
            "relevance": [
                "Does the text address the prompt?",
                "Is the content on-topic?",
                "Are there irrelevant tangents?"
            ],
            "informativeness": [
                "Does the text provide useful information?",
                "Is the content substantive?",
                "Are claims supported?"
            ]
        }


def demonstrate_generation_metrics():
    """Show how to evaluate generated text."""
    print("\n=== Text Generation Evaluation ===")

    # Example generated texts
    reference = "The quick brown fox jumps over the lazy dog"
    hypotheses = [
        "The quick brown fox jumps over the lazy dog",  # Perfect match
        "A fast brown fox leaps over a sleepy dog",  # Paraphrase
        "The fox jumps over the dog",  # Shorter
        "Something completely different here"  # Poor match
    ]

    print("Reference:", reference)
    print("\nHypotheses and BLEU scores:")
    for i, hyp in enumerate(hypotheses):
        bleu = GenerationMetrics.calculate_bleu_score(reference, hyp)
        print(f"{i + 1}. '{hyp}' - BLEU: {bleu:.3f}")

    # Diversity metrics
    generated_texts = [
        "The weather is nice today.",
        "The weather is great today.",
        "Today the weather is nice.",
        "It's a beautiful day outside.",
        "The sun is shining brightly."
    ]

    print("\n\nDiversity Metrics:")
    for n in [1, 2, 3]:
        diversity = GenerationMetrics.calculate_diversity(generated_texts, n=n)
        print(f"n={n}: {diversity}")

    # Human evaluation template
    print("\n\nHuman Evaluation Criteria:")
    template = GenerationMetrics.human_eval_template()
    for criterion, questions in template.items():
        print(f"\n{criterion.upper()}:")
        for q in questions:
            print(f"  - {q}")


# ============================================================================
# PRACTICAL LLM TIPS
# ============================================================================

def llm_interview_tips():
    """Practical tips for LLM-related interview questions."""
    print("\n=== LLM Interview Tips ===")

    tips = {
        "Architecture": [
            "Understand transformer architecture deeply",
            "Know the difference between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5)",
            "Be able to explain attention mechanism mathematically",
            "Understand positional encoding methods"
        ],

        "Training": [
            "Know about pre-training objectives (MLM, CLM, etc.)",
            "Understand fine-tuning vs prompt engineering trade-offs",
            "Be familiar with parameter-efficient methods (LoRA, QLoRA, Prefix Tuning)",
            "Know about training challenges (gradient accumulation, mixed precision)"
        ],

        "Inference": [
            "Understand different decoding strategies (greedy, beam search, sampling)",
            "Know about inference optimization (quantization, distillation)",
            "Be familiar with caching mechanisms for faster generation",
            "Understand memory requirements and optimization"
        ],

        "Evaluation": [
            "Know automatic metrics (BLEU, ROUGE, Perplexity) and their limitations",
            "Understand the importance of human evaluation",
            "Be familiar with benchmark datasets (GLUE, SuperGLUE, etc.)",
            "Know about bias and fairness evaluation"
        ],

        "Applications": [
            "Understand different prompting techniques",
            "Know about RAG (Retrieval-Augmented Generation)",
            "Be familiar with common use cases and their challenges",
            "Understand safety and alignment considerations"
        ]
    }

    for category, items in tips.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

    print("\n\nKey Concepts to Master:")
    print("1. Attention is All You Need - understand the paper")
    print("2. Scaling laws for neural language models")
    print("3. In-context learning and emergent abilities")
    print("4. Constitutional AI and RLHF")
    print("5. Prompt injection and security concerns")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all generative AI demonstrations."""
    print("=" * 60)
    print("DATA SCIENCE INTERVIEW PREP - GENERATIVE AI")
    print("=" * 60)

    # 1. Attention Mechanisms
    print("\n1. ATTENTION MECHANISMS")
    print("-" * 40)

    # Demo scaled dot-product attention
    seq_len, d_k = 5, 64
    query = np.random.randn(1, seq_len, d_k)
    key = np.random.randn(1, seq_len, d_k)
    value = np.random.randn(1, seq_len, d_k)

    output, weights = AttentionMechanisms.scaled_dot_product_attention(
        query, key, value, visualize=True
    )
    print(f"Attention output shape: {output.shape}")

    # 2. Positional Encoding
    print("\n\n2. POSITIONAL ENCODING")
    print("-" * 40)

    pe = PositionalEncoding.sinusoidal_encoding(50, 128, visualize=True)
    print(f"Positional encoding shape: {pe.shape}")

    # 3. Tokenization
    print("\n\n3. TOKENIZATION")
    print("-" * 40)

    demonstrate_tokenization()

    # 4. Prompt Engineering
    print("\n\n4. PROMPT ENGINEERING")
    print("-" * 40)

    demonstrate_prompting_techniques()

    # 5. Fine-tuning Setup
    print("\n\n5. FINE-TUNING")
    print("-" * 40)

    demonstrate_fine_tuning()

    # 6. Generation Metrics
    print("\n\n6. GENERATION EVALUATION")
    print("-" * 40)

    demonstrate_generation_metrics()

    # 7. Interview Tips
    print("\n\n7. INTERVIEW TIPS")
    print("-" * 40)

    llm_interview_tips()

    # 8. PyTorch Transformer Demo (if available)
    if TORCH_AVAILABLE:
        print("\n\n8. PYTORCH TRANSFORMER DEMO")
        print("-" * 40)

        # Create toy model
        vocab_size = 1000
        model = ToyGPT(vocab_size=vocab_size, d_model=256, n_heads=8,
                       n_layers=4, d_ff=1024, max_seq_len=128)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Toy GPT Parameters: {total_params:,}")

        # Example forward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")

    print("\n" + "=" * 60)
    print("GENERATIVE AI DEMONSTRATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Run demonstrations
    main()
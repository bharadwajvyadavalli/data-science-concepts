#!/usr/bin/env python3
"""
Data Science Interview Prep - Generative AI
==========================================
Key concepts and common interview questions
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

def attention_demo():
    """Demonstrate attention mechanism concept"""
    print("=== Attention Mechanism ===")
    
    # Simple attention example
    query = np.array([1, 2, 3])
    key = np.array([2, 1, 4])
    value = np.array([5, 6, 7])
    
    # Calculate attention scores
    scores = np.dot(query, key) / np.sqrt(len(query))
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Apply attention to values
    output = attention_weights * value
    
    print(f"Query: {query}")
    print(f"Key: {key}")
    print(f"Value: {value}")
    print(f"Attention weights: {attention_weights:.3f}")
    print(f"Output: {output:.3f}")
    print()

def multi_head_attention():
    """Explain multi-head attention"""
    print("=== Multi-Head Attention ===")
    print("1. Split input into multiple heads")
    print("2. Apply attention to each head independently")
    print("3. Concatenate results")
    print("4. Apply final linear transformation")
    print("Benefits:")
    print("- Allows model to focus on different aspects")
    print("- Captures different types of relationships")
    print("- Improves model capacity")
    print()

# ============================================================================
# TRANSFORMER ARCHITECTURE
# ============================================================================

def transformer_components():
    """Explain transformer components"""
    print("=== Transformer Components ===")
    print("1. Input Embedding + Positional Encoding")
    print("2. Multi-Head Self-Attention")
    print("3. Add & Norm (Residual Connection)")
    print("4. Feed-Forward Network")
    print("5. Add & Norm (Residual Connection)")
    print("6. Repeat for N layers")
    print()

def positional_encoding():
    """Demonstrate positional encoding"""
    print("=== Positional Encoding ===")
    
    # Simple sinusoidal encoding
    seq_len = 10
    d_model = 4
    
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    print("Positional encoding matrix:")
    print(pe)
    print()

# ============================================================================
# LANGUAGE MODELS
# ============================================================================

def language_model_concepts():
    """Explain key language model concepts"""
    print("=== Language Model Concepts ===")
    
    print("1. Tokenization:")
    print("   - Convert text to tokens")
    print("   - Subword tokenization (BPE, WordPiece)")
    print("   - Vocabulary size considerations")
    
    print("\n2. Training Objectives:")
    print("   - Masked Language Modeling (MLM)")
    print("   - Next Sentence Prediction (NSP)")
    print("   - Causal Language Modeling (CLM)")
    
    print("\n3. Model Types:")
    print("   - Encoder-only (BERT)")
    print("   - Decoder-only (GPT)")
    print("   - Encoder-Decoder (T5, BART)")
    print()

def prompt_engineering():
    """Demonstrate prompt engineering concepts"""
    print("=== Prompt Engineering ===")
    
    print("1. Zero-shot:")
    print("   Prompt: 'Classify sentiment: I love this movie'")
    print("   Expected: Positive")
    
    print("\n2. Few-shot:")
    print("   Prompt: 'Sentiment: I hate this -> Negative'")
    print("   'Sentiment: I love this -> Positive'")
    print("   'Sentiment: This is okay -> Neutral'")
    print("   'Sentiment: I love this movie'")
    
    print("\n3. Chain-of-Thought:")
    print("   Prompt: 'Let's solve step by step: If I have 3 apples...'")
    print()

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluation_metrics():
    """Explain evaluation metrics for language models"""
    print("=== Evaluation Metrics ===")
    
    print("1. Perplexity:")
    print("   - Lower is better")
    print("   - Measures how well model predicts next token")
    
    print("\n2. BLEU Score:")
    print("   - For machine translation")
    print("   - N-gram overlap with reference")
    
    print("\n3. ROUGE Score:")
    print("   - For text summarization")
    print("   - Recall-oriented metrics")
    
    print("\n4. Human Evaluation:")
    print("   - Fluency, coherence, relevance")
    print("   - Human preference scores")
    print()

# ============================================================================
# FINE-TUNING
# ============================================================================

def fine_tuning_concepts():
    """Explain fine-tuning concepts"""
    print("=== Fine-tuning Concepts ===")
    
    print("1. Full Fine-tuning:")
    print("   - Update all model parameters")
    print("   - Requires significant compute")
    print("   - Risk of catastrophic forgetting")
    
    print("\n2. Parameter-Efficient Fine-tuning:")
    print("   - LoRA (Low-Rank Adaptation)")
    print("   - AdaLoRA, QLoRA")
    print("   - Only update small number of parameters")
    
    print("\n3. Instruction Tuning:")
    print("   - Train on instruction-response pairs")
    print("   - Improves following instructions")
    print("   - Examples: Alpaca, Dolly")
    print()

# ============================================================================
# COMMON INTERVIEW QUESTIONS
# ============================================================================

def explain_attention():
    """Explain attention mechanism"""
    print("=== Attention Explanation ===")
    print("Attention allows model to focus on relevant parts of input")
    print("Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("Q: Query (what we're looking for)")
    print("K: Key (what we're matching against)")
    print("V: Value (what we're retrieving)")
    print()

def explain_vanishing_gradients():
    """Explain vanishing gradients in transformers"""
    print("=== Vanishing Gradients in Transformers ===")
    print("Problem: Gradients become very small in deep networks")
    print("Solutions in Transformers:")
    print("- Residual connections")
    print("- Layer normalization")
    print("- Multi-head attention")
    print("- Position-wise feed-forward networks")
    print()

def explain_scaling_laws():
    """Explain scaling laws"""
    print("=== Scaling Laws ===")
    print("Performance scales with:")
    print("- Model size (parameters)")
    print("- Training data size")
    print("- Compute budget")
    print("Chinchilla scaling: optimal model size ~ 20x data size")
    print()

def demonstrate_concepts():
    """Demonstrate key concepts"""
    print("=== Concept Demonstrations ===")
    
    # Simple text generation simulation
    print("1. Text Generation:")
    words = ["the", "cat", "sat", "on", "mat"]
    probabilities = [0.3, 0.2, 0.1, 0.2, 0.2]
    
    # Simulate next word prediction
    next_word = np.random.choice(words, p=probabilities)
    print(f"   Context: 'The cat'")
    print(f"   Predicted next word: '{next_word}'")
    
    # Attention visualization
    print("\n2. Attention Weights:")
    attention_matrix = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ])
    print(f"   Attention matrix:\n{attention_matrix}")
    print()

def main():
    """Run all demonstrations"""
    attention_demo()
    multi_head_attention()
    transformer_components()
    positional_encoding()
    language_model_concepts()
    prompt_engineering()
    evaluation_metrics()
    fine_tuning_concepts()
    explain_attention()
    explain_vanishing_gradients()
    explain_scaling_laws()
    demonstrate_concepts()

if __name__ == "__main__":
    main()
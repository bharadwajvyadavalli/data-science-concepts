# Data Science Interview Prep - Python Implementation Guide

This repository contains comprehensive Python implementations for data science technical interviews, organized by topic for easy navigation and study.

## 📁 File Structure

1. **`numpy_pandas.py`** - Python for Data Science
   - NumPy vectorization and broadcasting
   - Custom matrix operations (covariance, PCA)
   - Advanced Pandas operations
   - Data cleaning and EDA functions

2. **`statistics.py`** - Descriptive & Inferential Statistics
   - Descriptive statistics from scratch
   - Statistical tests (t-test, ANOVA, chi-squared)
   - Probability distributions
   - Bootstrap and confidence intervals
   - Central Limit Theorem simulation

3. **`machine_learning.py`** - Statistical Learning & ML
   - Linear/Logistic Regression from scratch
   - Decision Trees implementation
   - K-Means clustering
   - Cross-validation and hyperparameter tuning
   - Classification metrics and evaluation

4. **`deep_learning.py`** - Deep Learning
   - Neural Network from scratch with backpropagation
   - PyTorch CNN for MNIST
   - Custom datasets and data loaders
   - Learning rate scheduling
   - Early stopping and checkpointing

5. **`generative_ai.py`** - Generative AI & Foundation Models
   - Toy GPT implementation with attention
   - Tokenization and positional encoding
   - Prompt engineering examples
   - Fine-tuning with HuggingFace
   - Generation evaluation metrics

## 🚀 How to Use This Guide

### For Interview Preparation:

1. **Start with Fundamentals**: Begin with `01_numpy_pandas.py` to ensure strong foundation
2. **Implement First**: Try coding algorithms yourself before looking at solutions
3. **Understand the Math**: Make sure you can explain each step
4. **Practice Variations**: Modify parameters and test edge cases
5. **Time Yourself**: Practice implementing key algorithms within interview time constraints

### During Interviews:

1. **Clarify Requirements**: Always ask about input format, edge cases, and expected output
2. **Start Simple**: Begin with brute force, then optimize
3. **Think Aloud**: Explain your reasoning as you code
4. **Test Your Code**: Run through examples and edge cases
5. **Discuss Complexity**: Mention time and space complexity

## 💡 Key Interview Tips

### Problem-Solving Approach:
```python
# 1. Understand the problem
# 2. Work through examples
# 3. Code a simple solution
# 4. Test and debug
# 5. Optimize if needed
# 6. Analyze complexity
```

### Code Organization:
```python
# Always structure your code clearly:
import necessary_libraries

def helper_function():
    """Document your functions"""
    pass

class YourSolution:
    """Use classes for complex implementations"""
    
    def __init__(self):
        # Initialize parameters
        pass
    
    def main_method(self):
        # Core logic here
        pass

# Test with examples
if __name__ == "__main__":
    # Test your implementation
    pass
```

### Common Pitfalls to Avoid:
- Forgetting edge cases (empty arrays, None values)
- Numerical instability (overflow, division by zero)
- Inefficient nested loops when vectorization is possible
- Not validating input assumptions
- Overcomplicating simple problems

## 📊 Quick Reference - What to Know

### Must-Know Implementations:
- [ ] Linear/Logistic Regression gradient descent
- [ ] Decision Tree splitting criteria
- [ ] K-Means clustering steps
- [ ] Neural network forward/backward pass
- [ ] Cross-validation from scratch
- [ ] Basic statistical tests

### Key Concepts to Explain:
- [ ] Bias-variance tradeoff
- [ ] Overfitting and regularization
- [ ] Gradient descent variants
- [ ] Evaluation metrics (precision, recall, F1, AUC)
- [ ] Feature engineering techniques
- [ ] Attention mechanism basics

## 🔧 Environment Setup

```bash
# Required libraries
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install torch torchvision  # For deep learning
pip install transformers datasets  # For NLP/GenAI

# Optional but recommended
pip install jupyter notebook  # For interactive development
pip install pytest  # For testing your implementations
```

## 📈 Performance Benchmarks

When implementing algorithms, aim for these performance targets:

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Linear Regression | O(n²p) training | O(np) |
| Decision Tree | O(n²p) training | O(nodes) |
| K-Means | O(nkdi) | O(nk) |
| Neural Network | O(n·m·h·e) | O(parameters) |

Where: n=samples, p=features, k=clusters, d=dimensions, i=iterations, m=batch_size, h=hidden_units, e=epochs

## 🎯 Interview Success Checklist

Before your interview, make sure you can:

- [ ] Implement key algorithms without looking at references
- [ ] Explain the math behind each algorithm
- [ ] Discuss pros/cons and when to use each method
- [ ] Handle edge cases gracefully
- [ ] Optimize for both time and space complexity
- [ ] Write clean, readable code quickly
- [ ] Debug efficiently when things go wrong

## 📚 Additional Resources

- Review probability and linear algebra basics
- Practice on platforms like LeetCode (for coding) and Kaggle (for ML)
- Mock interview with peers
- Review recent papers for GenAI topics

---

**Remember**: The goal isn't just to memorize implementations, but to understand them deeply enough to adapt and explain them in any interview scenario. Good luck! 🚀
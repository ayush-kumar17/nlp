# NLP Assignment 1 – GloVe Implementation from Scratch

## Overview

This project presents an implementation of the GloVe (Global Vectors for Word Representation) algorithm built entirely from scratch using NumPy and SciPy. The goal of this assignment is to understand how word embeddings are learned through global co-occurrence statistics and how optimization is performed using the AdaGrad algorithm.

The implementation includes:

- Construction of a word co-occurrence matrix
- Implementation of the weighted GloVe objective function
- Training using AdaGrad optimization
- Experiments with multiple embedding dimensions
- Tracking of training loss and runtime
- Saving embeddings and training summaries

This project focuses on understanding the mathematical foundations and implementation details of GloVe rather than using high-level deep learning libraries.

---

## The GloVe Model

GloVe is a count-based model that learns word vectors by factorizing a global word-word co-occurrence matrix.

The objective function minimized during training is:

J = Σᵢⱼ f(Xᵢⱼ) (wᵢᵀ wⱼ + bᵢ + bⱼ − log(Xᵢⱼ))²

Where:

- Xᵢⱼ is the co-occurrence count between words i and j
- wᵢ and wⱼ are the word and context vectors
- bᵢ and bⱼ are bias terms
- f(x) is a weighting function that reduces the impact of very frequent co-occurrences

The weighting function used in this implementation is:

f(x) = (x / X_max)^α  if x < X_max  
f(x) = 1              otherwise

Hyperparameters:
- X_max = 100
- α = 0.75

---

## Hyperparameters Used

WINDOW_SIZE = 10  
LEARNING_RATE = 0.05  
NUM_ITERATIONS = 50  
EMBEDDING_DIMS = [50, 100, 200, 300]  
MAX_DOCUMENTS = 67093  

---

## Project Structure

nlp_assignment1/

data/  
  updated_vocab_document_dict.json  

outputs/  
  cooccurrence_w10.pkl  
  glove_embeddings_d50.npy  
  glove_embeddings_d100.npy  
  glove_embeddings_d200.npy  
  glove_embeddings_d300.npy  
  training_summary.pkl  
  loss_histories.pkl  
  loss_curves.png  

task1_pretraining.py  
task1_tuning.py  
task1_analysis.py  
requirements.txt  
README.md  

---

## Implementation Details

### 1. Document Processing

Documents are reconstructed from the vocabulary-document dictionary. Text is tokenized using regular expressions, and words are mapped to indices using a vocabulary dictionary.

### 2. Co-occurrence Matrix Construction

A sliding window approach (window size = 10) is used to compute word co-occurrences. Context words are weighted by inverse distance from the center word. The matrix is stored using a sparse representation (LIL format) and converted to CSR format for efficient computation.

### 3. Model Initialization

For each embedding dimension:
- Word vectors and context vectors are initialized randomly
- Bias terms are initialized to zero
- AdaGrad accumulators are initialized to ones

### 4. Training

For each epoch:
- Co-occurrence pairs are shuffled
- The weighted squared loss is computed
- Gradients are calculated
- Parameters are updated using AdaGrad
- Average loss per epoch is recorded

Final word vectors are computed as:

(W + Wc) / 2

---

## Outputs

For each embedding dimension:
- Trained word embeddings (.npy files)
- Training loss history
- Training time

Additionally:
- training_summary.pkl contains hyperparameters and timing information
- loss_histories.pkl stores loss curves and runtime data

---

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run training:

python task1_pretraining.py

---

## Observations

- Larger embedding dimensions increase training time significantly.
- The loss decreases consistently across epochs, indicating proper optimization.
- Building the co-occurrence matrix is the most memory-intensive step.

---

## Conclusion

This project demonstrates a complete implementation of the GloVe algorithm from scratch. It provides a detailed understanding of:

- How global co-occurrence statistics are computed
- How the GloVe objective function is derived and optimized
- How AdaGrad adapts learning rates during training
- How embedding dimensionality affects training time and performance

The implementation reinforces both theoretical understanding and practical skills in natural language processing.

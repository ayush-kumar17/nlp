import numpy as np
import json
import pickle
import re
import time
import os
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_DIM = 200
FIXED_ITERATIONS = 20
MAX_DOCUMENTS = 67093
X_MAX = 100
ALPHA = 0.75

WINDOW_SIZES = [5, 10, 15]
LEARNING_RATES = [0.01, 0.05, 0.1]

print(f"Fixed dimension: {FIXED_DIM}")
print(f"Iterations: {FIXED_ITERATIONS}")
print(f"Documents: {MAX_DOCUMENTS}")
print(f"Testing {len(WINDOW_SIZES)} windows {len(LEARNING_RATES)} learning rates")

with open(os.path.join(DATA_DIR, "updated_vocab_document_dict.json")) as f:
    vocab_data = json.load(f)

vocab = list(vocab_data.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocabulary size: {len(vocab)}")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def get_documents(vocab_data, max_docs):
    docs = defaultdict(list)
    for word, passages in vocab_data.items():
        for doc_id, text in passages:
            if doc_id < max_docs:
                docs[doc_id].append(text)
    result = [" ".join(docs[i]) for i in sorted(docs)]
    print(f"Extracted {len(result)} documents")
    return result

def build_cooccurrence(docs, window, vocab, word_to_idx):
    print(f"Building cooccurrence matrix with window={window}")
    V = len(vocab)
    cooc = lil_matrix((V, V), dtype=np.float32)
    
    for doc in tqdm(docs, desc="Processing docs"):
        tokens = [word_to_idx[w] for w in tokenize(doc) if w in word_to_idx]
        for i, wi in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    cooc[wi, tokens[j]] += 1.0 / abs(i - j)
    
    cooc = cooc.tocsr()
    print(f"Non-zero entries: {cooc.nnz:,}")
    return cooc

class GloVe:
    def __init__(self, vocab_size, dim):
        self.dim = dim
        scale = 0.5 / dim
        self.W = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)
        self.Wc = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)
        self.b = np.zeros(vocab_size, dtype=np.float32)
        self.bc = np.zeros(vocab_size, dtype=np.float32)
        self.gW = np.ones_like(self.W)
        self.gWc = np.ones_like(self.Wc)
        self.gb = np.ones_like(self.b)
        self.gbc = np.ones_like(self.bc)
    
    def weight(self, x):
        return (x / X_MAX) ** ALPHA if x < X_MAX else 1.0
    
    def train(self, cooc, lr, iterations):
        coo = cooc.tocoo()
        pairs = list(zip(coo.row, coo.col, coo.data))
        losses = []
        
        for it in range(iterations):
            np.random.shuffle(pairs)
            total_loss = 0.0
            
            for i, j, x in tqdm(pairs, desc=f"Epoch {it+1}", leave=False):
                w = self.weight(x)
                diff = np.dot(self.W[i], self.Wc[j]) + self.b[i] + self.bc[j] - np.log(x)
                total_loss += w * diff * diff
                
                grad = w * diff
                dW = grad * self.Wc[j]
                dWc = grad * self.W[i]
                
                self.gW[i] += dW ** 2
                self.gWc[j] += dWc ** 2
                self.gb[i] += grad ** 2
                self.gbc[j] += grad ** 2
                
                self.W[i] -= lr * dW / np.sqrt(self.gW[i])
                self.Wc[j] -= lr * dWc / np.sqrt(self.gWc[j])
                self.b[i] -= lr * grad / np.sqrt(self.gb[i])
                self.bc[j] -= lr * grad / np.sqrt(self.gbc[j])
            
            avg_loss = total_loss / len(pairs)
            losses.append(avg_loss)
            print(f"Epoch {it+1}: loss={avg_loss:.6f}")
        
        return losses

docs = get_documents(vocab_data, MAX_DOCUMENTS)

results = []

for window in WINDOW_SIZES:
    cooc = build_cooccurrence(docs, window, vocab, word_to_idx)
    
    for lr in LEARNING_RATES:
        print(f"\nTesting window={window}, lr={lr}")
        model = GloVe(len(vocab), FIXED_DIM)
        t_start = time.time()
        losses = model.train(cooc, lr, FIXED_ITERATIONS)
        t_elapsed = time.time() - t_start
        
        results.append({'window': window,'lr': lr,'final_loss': losses[-1],'losses': losses,'time': t_elapsed })
        print(f"Final loss: {losses[-1]:.6f}, time: {t_elapsed:.1f}s")

results.sort(key=lambda x: x['final_loss'])
best = results[0]

print("\nResults")
print(f"{'Rank':<6} {'Window':<8} {'LR':<8} {'Loss':<12} {'Time(s)':<10}")
for rank, r in enumerate(results, 1):
    print(f"{rank:<6} {r['window']:<8} {r['lr']:<8.2f} {r['final_loss']:<12.6f} {r['time']:<10.1f}")

print(f"\nBest config: window={best['window']}, lr={best['lr']}, loss={best['final_loss']:.6f}")

with open(os.path.join(OUTPUT_DIR, "tuning_results.txt"), 'w') as f:
    f.write("Hyperparameter Tuning Results\n\n")
    f.write(f"Best Configuration:\n")
    f.write(f"  Window size: {best['window']}\n")
    f.write(f"  Learning rate: {best['lr']}\n")
    f.write(f"  Final loss: {best['final_loss']:.6f}\n\n")
    f.write("All Results:\n")
    for rank, r in enumerate(results, 1):
        f.write(f"{rank}. window={r['window']}, lr={r['lr']:.2f}, loss={r['final_loss']:.6f}\n")

best_config = {
    'window_size': best['window'],
    'learning_rate': best['lr']
}
with open(os.path.join(OUTPUT_DIR, "best_config.pkl"), 'wb') as f:
    pickle.dump(best_config, f)

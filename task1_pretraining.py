import numpy as np
import json
import pickle
import re
import os
import time
from collections import defaultdict
from scipy.sparse import lil_matrix
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 10
LEARNING_RATE = 0.05
NUM_ITERATIONS = 50
X_MAX = 100
ALPHA = 0.75
EMBEDDING_DIMS = [50, 100, 200, 300]
MAX_DOCUMENTS = 67093

print(f"Window: {WINDOW_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Dimensions: {EMBEDDING_DIMS}")
print(f"Max documents: {MAX_DOCUMENTS}")

with open(os.path.join(DATA_DIR, "updated_vocab_document_dict.json")) as f:
    vocab_data = json.load(f)

vocab = list(vocab_data.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocab size: {len(vocab)}")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def get_documents(vocab_data, max_docs):
    docs = defaultdict(list)
    for word, passages in vocab_data.items():
        for doc_id, text in passages:
            if doc_id < max_docs:
                docs[doc_id].append(text)
    result = [" ".join(docs[i]) for i in sorted(docs)]
    print(f"Got {len(result)} documents")
    return result

def build_cooccurrence(docs, window):
    print(f"Building cooccurrence matrix with window={window}")
    V = len(vocab)
    cooc = lil_matrix((V, V), dtype=np.float32)
    
    for doc in tqdm(docs):
        tokens = [word_to_idx[w] for w in tokenize(doc) if w in word_to_idx]
        for i, wi in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    cooc[wi, tokens[j]] += 1.0 / abs(i - j)
    
    cooc = cooc.tocsr()
    print(f"Done, non-zero entries: {cooc.nnz:,}")
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
    
    def vectors(self):
        return (self.W + self.Wc) / 2

docs = get_documents(vocab_data, MAX_DOCUMENTS)

cooc_file = os.path.join(OUTPUT_DIR, f"cooccurrence_w{WINDOW_SIZE}.pkl")
if os.path.exists(cooc_file):
    print(f"Loading cooccurrence from {cooc_file}")
    with open(cooc_file, 'rb') as f:
        cooc = pickle.load(f)
    print("Loaded")
else:
    cooc = build_cooccurrence(docs, WINDOW_SIZE)
    with open(cooc_file, 'wb') as f:
        pickle.dump(cooc, f)
    print(f"Saved cooccurrence to {cooc_file}")

all_losses = {}
all_times = {}

for d in EMBEDDING_DIMS:
    print(f"\nTraining d={d}")
    model = GloVe(len(vocab), d)
    start = time.time()
    losses = model.train(cooc, LEARNING_RATE, NUM_ITERATIONS)
    elapsed = time.time() - start
    
    all_losses[d] = losses
    all_times[d] = elapsed
    
    emb_file = os.path.join(OUTPUT_DIR, f"glove_embeddings_d{d}.npy")
    np.save(emb_file, model.vectors())
    print(f"Saved to {emb_file}")
    print(f"Time: {elapsed/60:.1f} min")

training_summary = {
    'hyperparameters': {
        'window_size': WINDOW_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_iterations': NUM_ITERATIONS,
        'x_max': X_MAX,
        'alpha': ALPHA
    },
    'embedding_dims': EMBEDDING_DIMS,
    'loss_histories': all_losses,
    'training_times': all_times,
    'vocab_size': len(vocab),
    'num_documents': MAX_DOCUMENTS
}

with open(os.path.join(OUTPUT_DIR, "training_summary.pkl"), 'wb') as f:
    pickle.dump(training_summary, f)

with open(os.path.join(OUTPUT_DIR, "loss_histories.pkl"), 'wb') as f:
    pickle.dump({'losses': all_losses, 'times': all_times}, f)

print(f"\nTotal time: {sum(all_times.values())/60:.1f} min")
for d in EMBEDDING_DIMS:
    print(f"  glove_embeddings_d{d}.npy: {all_times[d]/60:.1f} min")
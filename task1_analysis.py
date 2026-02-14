import numpy as np
import json
import pickle
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR="data"
OUTPUT_DIR="outputs"

EMBEDDING_DIMS=[50, 100, 200, 300]
TEST_WORDS=['economy', 'president', 'running']

with open(os.path.join(DATA_DIR, "updated_vocab_document_dict.json")) as f:
    vocab_data = json.load(f)

vocab = list(vocab_data.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
with open(os.path.join(OUTPUT_DIR, "training_summary.pkl"), 'rb') as f:
    training_summary = pickle.load(f)

all_losses= training_summary['loss_histories']
all_times= training_summary['training_times']
best_config = {
    'window_size': training_summary['hyperparameters']['window_size'],
    'learning_rate': training_summary['hyperparameters']['learning_rate']
}

embeddings = {}
for d in EMBEDDING_DIMS:
    embeddings[d] = np.load(os.path.join(OUTPUT_DIR, f"glove_embeddings_d{d}.npy"))
print("Create loss plot")

plt.figure(figsize=(10, 6))

for d, losses in all_losses.items():
    plt.plot(range(1, len(losses)+1), losses, 
             marker='o', markersize=3, label=f'd={d}', linewidth=2)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('GloVe Training Loss for Different Dimensions', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file = os.path.join(OUTPUT_DIR, 'loss_curves.png')
plt.savefig(plot_file, dpi=300)
print(f"Saved {plot_file}")
plt.close()
print("\nFinding nearest neighbors (d=200)")

def get_similar_words(embeddings, word, k=5):
    if word not in word_to_idx:
        return []
    
    idx = word_to_idx[word]
    word_vec = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(word_vec, embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:k+1]
    
    results = []
    for i in top_idx:
        if i != idx:
            results.append((vocab[i], sims[i]))
        if len(results) == k:
            break
    return results

emb_200 = embeddings[200]

for word in TEST_WORDS:
    print(f"\nWord: '{word}'")
    neighbors = get_similar_words(emb_200, word, k=5)
    for rank, (w, score) in enumerate(neighbors, 1):
        print(f"  {rank}. {w:20s} ({score:.4f})")

print("\nWriting summary file")

txt_file = os.path.join(OUTPUT_DIR, 'training_summary.txt')
with open(txt_file, 'w') as f:
    f.write("GloVe training results\n\n")
    
    f.write("Configuration:\n")
    f.write(f"Window size: {best_config['window_size']}\n")
    f.write(f"Learning rate: {best_config['learning_rate']}\n")
    f.write(f"Training iterations: 50\n")
    f.write(f"X_max: 100, Alpha: 0.75\n")
    f.write(f"Embedding dimensions: {EMBEDDING_DIMS}\n")
    f.write(f"Documents: {training_summary['num_documents']:,}\n")
    f.write(f"Vocabulary: {len(vocab):,}\n\n")
    
    f.write("Training Results:\n")
    f.write(f"{'Dimension':<12} {'Training Time':<20} {'Final Loss':<15}\n")
    for d in EMBEDDING_DIMS:
        time_str = f"{all_times[d]/60:.1f} min"
        loss_str = f"{all_losses[d][-1]:.6f}"
        f.write(f"d={d:<9} {time_str:<20} {loss_str:<15}\n")
    f.write("\n")
    
    f.write("Nearest Neighbors (d=200):\n")
    for word in TEST_WORDS:
        f.write(f"\nWord: '{word}'\n")
        neighbors = get_similar_words(emb_200, word, k=5)
        for rank, (w, score) in enumerate(neighbors, 1):
            f.write(f"  {rank}. {w:20s} ({score:.4f})\n")

print(f"Saved {txt_file}")

print("Creating CSV")

csv_file = os.path.join(OUTPUT_DIR, 'loss_data.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Iteration'] + [f'd={d}' for d in EMBEDDING_DIMS]
    writer.writerow(header)
    
    for i in range(50):
        row = [i+1] + [all_losses[d][i] for d in EMBEDDING_DIMS]
        writer.writerow(row)

print(f"Saved {csv_file}")

summary_data = {
    'configuration': {'window_size': best_config['window_size'],'learning_rate': best_config['learning_rate'],'iterations': 50,'x_max': 100,'alpha': 0.75,'dimensions': EMBEDDING_DIMS,
        'vocabulary_size': len(vocab),'documents': training_summary['num_documents']},
    'results': {
        f'd{d}': {
            'training_time_minutes': round(all_times[d]/60, 2),'final_loss': float(all_losses[d][-1])
        } for d in EMBEDDING_DIMS
    }
}

json_file = os.path.join(OUTPUT_DIR, 'training_summary.json')
with open(json_file, 'w') as f:
    json.dump(summary_data, indent=2, fp=f)

print(f"Saved {json_file}")
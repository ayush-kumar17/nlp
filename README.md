# Task 1: GloVe Pre-training

## Setup Complete ✓

### Data Files Needed
Place these files in the `data/` folder:
- `vocabulary.txt` (your vocab file)
- `ccnews_subset.json` (your CC-News data)

### Run Task 1
```bash
source venv/bin/activate
python task_1.py
```

### Configuration
Edit `Config` class in `task_1.py`:
- WINDOW_SIZE: 10 (try 5, 10, 15)
- LEARNING_RATE: 0.05 (try 0.01, 0.05, 0.1)
- NUM_ITERATIONS: 50
- EMBEDDING_DIMS: [50, 100, 200, 300]

### Outputs
All results in `outputs/`:
- Co-occurrence matrices (.pkl)
- Word embeddings (.npy)
- Loss curves (.png)
- Training summary (.pkl)

### Deactivate
```bash
deactivate
```
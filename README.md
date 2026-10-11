# Bach Chorale Generation system using RNNs (LSTMs)

An end-to-end Generative AI pipeline in TensorFlow/Keras to parse, tokenize, preprocess, train, and generate original Bach-style chorales using stacked Long Short-Term Memory (LSTM) networks.

## Project Architecture
This project is organized into 7 distinct phases across modular scripts:
1. **Phase 1: Data Collection & Tokenization (`data_loader.py`)** - Fetches and parses music scores from `music21`, building a bidirectional vocabulary map.
2. **Phase 2: Sequence Preparation (`preprocess.py`)** - Prepares sequence windows of length 100 with train/val partitioning.
3. **Phase 3: Model Architecture Setup (`model.py`)** - Implements a stacked LSTM neural network with Dropout.
4. **Phase 4: Model Training (`train.py`)** - Manages model training, learning rate scheduling, checkpoints, and saves loss curves.
5. **Phase 5: Music Generation (`generate.py`)** - Implements temperature-based sampling for discrete tokens.
6. **Phase 6: MIDI Export (`export.py`)** - Converts generated sequences back to MIDI streams.
7. **Phase 7: Project Evaluation & Portfolio** - Establishes heuristic metrics and professional deliverables.

## Installation & Setup
```bash
pip install -r requirements.txt
```

## Running the Pipeline
- To download data and tokenize: Run `data_loader.py`.
- To train the model: Run `train.py`.
- To generate music: Run `generate.py`.
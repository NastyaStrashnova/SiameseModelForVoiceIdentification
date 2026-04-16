# Siamese Neural Network for Voice Authentication
## Overview

This project implements a voice authentication system using a Siamese Neural Network trained on spectrogram representations of audio signals. The model learns to determine whether two audio samples belong to the same speaker by comparing learned embeddings.

The full pipeline—from raw audio preprocessing to model training and evaluation—is implemented in a single, structured Jupyter notebook.

## Problem Statement

Voice authentication requires learning speaker-specific features that remain robust across different recordings. Instead of directly classifying speakers, this project approaches the problem as similarity learning, where the model predicts whether two samples belong to the same identity.

## Approach

The system follows a multi-stage pipeline:

## 1. Audio Preprocessing
Conversion from .flac to .wav
Silence trimming using librosa
Fixed-length signal extraction for consistency
## 2. Feature Extraction
Spectrogram generation from audio signals
Transformation of time-series audio into image-like representations suitable for CNNs
## 3. Dataset Construction
Triplet-style data organization:
Anchor (reference sample)
Positive (same speaker)
Negative (different speaker)
Built using tf.data with:
caching
shuffling
batching
prefetching
## 4. Model Architecture
Embedding Network
Convolutional Neural Network (CNN)
Multiple convolution + pooling blocks
Final dense layer producing a 4096-dimensional embedding
Siamese Network
Shared embedding network for both inputs
L1 distance layer to measure similarity
Binary classifier to predict match / non-match
## 5. Training
- Custom training loop using tf.GradientTape
- Loss: Binary Crossentropy
- Optimizer: Adam
- Checkpointing for training stability
## 6. Evaluation

The model is evaluated using:

Precision
Recall
Accuracy
Equal Error Rate (EER), a standard metric for biometric systems
Results
Recall: ~0.99
Precision: ~0.94
Accuracy: ~0.96
Equal Error Rate (EER): ~0.045

These results demonstrate strong performance in distinguishing between same-speaker and different-speaker pairs.

Tech Stack
Python
TensorFlow / Keras
Librosa (audio processing)
NumPy / SciPy
tf.data (data pipeline)
Repository Structure

This project is implemented as a single Jupyter notebook, which includes:

Data preprocessing and spectrogram generation
Dataset pipeline construction
Model definition (embedding + Siamese network)
Custom training loop
Evaluation and metrics

👉 The full implementation is available here:
[GitHub Repository Link]

Key Takeaways
Demonstrates deep learning for similarity learning (Siamese networks)
Implements a complete ML pipeline from raw data to evaluation
Uses custom training logic and structured data pipelines
Focuses on robust feature learning and model generalization

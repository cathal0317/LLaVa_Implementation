# Speech-to-LLaMA: A LLaVA-Style Implementation for ASR

This repository contains a simple LLaVA-style implementation in the speech domain.  
The project explores how a pretrained speech encoder can be aligned with a pretrained language model using a lightweight projection module for automatic speech recognition.

The current setup combines a Whisper encoder with a TinyLlama language model. Speech representations are projected into the LLM embedding space and used as a prefix for text generation.

## Overview

Inspired by LLaVA, this project applies the encoder–projector–LLM paradigm to speech.

The pipeline is:

1. Encode audio using Whisper
2. Reduce sequence length via pooling
3. Project speech features into the LLM hidden space
4. Concatenate projected speech embeddings with text token embeddings
5. Train the model to generate transcripts

This is a small-scale implementation intended for experimentation and understanding multimodal alignment, rather than a fully optimised system.

## Model

The model consists of:

- **Speech encoder:** `openai/whisper-base`
- **Language model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Projection module:** two-layer MLP with GELU activation

Whisper produces frame-level hidden states, which are downsampled and projected into the LLaMA hidden dimension. These projected embeddings are prepended to token embeddings and passed into the language model.

## Dataset

Experiments are currently conducted on the **LibriSpeech ASR (clean split)** using the Hugging Face `datasets` library.

Each sample contains:
- audio waveform
- corresponding transcript


## Installation

```bash
git clone https://github.com/cathal0317/LLaVa_Implementation.git
cd LLaVa_Implementation

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt



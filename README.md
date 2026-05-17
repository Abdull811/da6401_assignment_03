# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

This project implements the Transformer architecture proposed in the landmark paper *“Attention Is All You Need”* using PyTorch. The objective is to build a Neural Machine Translation (NMT) system capable of translating German sentences into English using the Multi30k dataset.

The implementation includes the complete Transformer pipeline from scratch, including:

- Multi-Head Self Attention
- Positional Encoding
- Encoder–Decoder Architecture
- Scaled Dot-Product Attention
- Noam Learning Rate Scheduler
- Label Smoothing
- Greedy Decoding for Inference

In addition to the implementation, several ablation studies and experiments were performed to analyze important Transformer design choices such as:
- Noam Scheduler vs Fixed Learning Rate
- Scaling Factor in Attention
- Attention Head Specialization
- Positional Encoding vs Learned Embeddings
- Label Smoothing Effects

Interactive experiment tracking, plots, gradient analysis, and attention visualizations were logged using Weights & Biases (W&B).

---

## W&B Report

W&B Report Link:  
https://wandb.ai/ge26z811-zan/da6401_Assigment_03_Weight_&_Bias/reports/DA6401-Assignment-3-Implementing-the-Transformer-for-Machine-Translation--VmlldzoxNjkxMjE5OA

---

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Transformer architecture (Encoder, Decoder, Multi-Head Attention)
├── utils.py           # Noam Scheduler, Label Smoothing, Masking Utilities
├── dataset.py         # Multi30k dataset loading and tokenization using spaCy
├── train.py           # Training loop, evaluation, and greedy decoding inference

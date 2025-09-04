# Protein-Ligand Affinity Prediction with Hybrid GNN

This repository contains a hybrid model for predicting the binding affinity between proteins and ligands using **Graph Neural Networks (GNN)** and **Transformer-based embeddings** (ProtBERT for proteins). The dataset was sourced from the Kaggle competition [Structure-free protein-ligand affinity prediction](https://kaggle.com/competitions/protein-compound-affinity).

## What is Binding Affinity?

Binding affinity, often measured by **IC50** (**half-maximal inhibitory concentration**), refers to the strength of the interaction between a protein and a ligand. The lower the **IC50** value, the stronger the binding affinity. In simple terms, we want to predict how tightly a ligand binds to a protein based on the protein sequence and ligand structure.

## Model Architecture

The model integrates:

- **ProtBERT**: A pre-trained transformer model from **Hugging Face** to encode protein sequences into embeddings.
- **Ligand Graph Construction**: Ligand SMILES strings are converted into molecular graphs, where nodes represent atoms and edges represent bonds.
- **Hybrid GNN**: A combination of **GINConv**, **GATConv**, and **SAGEConv** layers processes the ligand graphs.
- **Embedding Fusion**: The outputs from the protein embeddings (ProtBERT) and ligand graph embeddings (GNN) are concatenated to predict binding affinity.

## How It Works

By combining **ProtBERT** (for protein sequence embeddings) with graph-based models (for ligand structure), the model learns to predict how well a protein and ligand interact, which is represented by the **IC50** value.

## Training & Results

- **Dataset**: Trained on **100k samples**.
- **Performance**: The model achieved an **MSE of 1.23**

## Requirements

### Software:
- **Python 3.8+**

### Libraries:
- **PyTorch**: The deep learning framework for building and training the model.
- **PyTorch Geometric**: For implementing graph-based neural networks.
- **Hugging Face Transformers**: For the pre-trained ProtBERT model to encode protein sequences.
- **Pandas**: For handling datasets and data manipulation.
- **NumPy**: For numerical operations and array handling.
- **Scikit-learn**: For data preprocessing and model evaluation.

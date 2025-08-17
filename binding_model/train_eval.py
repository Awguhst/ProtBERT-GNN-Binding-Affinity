import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import r2_score, mean_squared_error

# Define the training function
def train(loader, model, criterion, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = total_samples = 0

    # Iterate through batches in the DataLoader
    for protein_embedding, ligand_graph, labels_batch in loader:
        # Move data to the appropriate device
        protein_embedding = protein_embedding.to(device)
        ligand_graph = ligand_graph.to(device)  # This should be a graph (e.g., Data object)
        labels_batch = labels_batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        # The model expects the following:
        #   ligand_graph.x (node features), ligand_graph.edge_index (edge list), ligand_graph.batch (batch indices), and protein_embedding
        affinity_pred = model(ligand_graph.x, ligand_graph.edge_index, ligand_graph.batch, protein_embedding)

        # Compute MSE loss
        loss = criterion(affinity_pred, labels_batch)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate weighted loss and total samples
        total_loss += loss.detach().item() * len(labels_batch)
        total_samples += len(labels_batch)

    # Return root mean squared error (RMSE) for training
    return np.sqrt(total_loss / total_samples)


# Define the validation function
@torch.no_grad()  # Disable gradient tracking for validation (improves speed and reduces memory)
def val(loader, model, criterion, device):
    model.eval()  # Set model to evaluation mode
    
    total_loss = total_samples = 0

    # Iterate through the validation dataset
    for protein_embedding, ligand_graph, labels_batch in loader:
        # Move data to the appropriate device
        protein_embedding = protein_embedding.to(device)
        ligand_graph = ligand_graph.to(device)  # This should be a graph (e.g., Data object)
        labels_batch = labels_batch.to(device)

        # Forward pass through the model
        # The model expects the following:
        #   ligand_graph.x (node features), ligand_graph.edge_index (edge list), ligand_graph.batch (batch indices), and protein_embedding
        affinity_pred = model(ligand_graph.x, ligand_graph.edge_index, ligand_graph.batch, protein_embedding)

        # Compute MSE loss
        loss = criterion(affinity_pred, labels_batch)

        # Accumulate weighted loss and total samples
        total_loss += loss.detach().item() * len(labels_batch)
        total_samples += len(labels_batch)

    # Compute and return RMSE for validation
    return np.sqrt(total_loss / total_samples)

@torch.no_grad()  # Disable gradient tracking for evaluation (improves speed and reduces memory)
def evaluate(loader, model, device):
    model.eval()  # Set model to evaluation mode
    
    all_preds = []
    all_labels = []

    # Iterate through the validation dataset
    for protein_embedding, ligand_graph, labels_batch in loader:
        # Move data to the appropriate device
        protein_embedding = protein_embedding.to(device)
        ligand_graph = ligand_graph.to(device)  # This should be a graph (e.g., Data object)
        labels_batch = labels_batch.to(device)

        affinity_pred = model(ligand_graph.x, ligand_graph.edge_index, ligand_graph.batch, protein_embedding)

        # Store predictions and labels for metric calculation
        all_preds.append(affinity_pred.cpu().numpy())  # Move to CPU and convert to numpy
        all_labels.append(labels_batch.cpu().numpy())

    # Concatenate all predictions and labels into single arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate R², MSE, and RMSE
    r2 = r2_score(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)

    # Print the metrics directly
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

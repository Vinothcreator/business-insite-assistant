# train_conversation_assistant.py
import numpy as np
import pandas as pd
import json
import os

# Paths
WORKSPACE_DIR = r"D:\Data_Analysatics\Mini-project4"
METADATA_CSV = os.path.join(WORKSPACE_DIR, "saas_sales_conversations.csv")
EMBEDDINGS_JSON = os.path.join(WORKSPACE_DIR, "saas_sales_embeddings.json")
MODEL_OUTPUT_JSON = os.path.join(WORKSPACE_DIR, "saas_sales_model_weights.json")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def train_model(epochs=100, learning_rate=0.1, test_split=0.2):
    print("=" * 60)
    print("TRAINING CONVERSATION AI ASSISTANT MODEL (Pure NumPy)")
    print("=" * 60)
    
    # 1. Load data
    if not os.path.exists(METADATA_CSV) or not os.path.exists(EMBEDDINGS_JSON):
        raise FileNotFoundError("Dataset files not found. Please run generate_conversation_dataset.py first.")
        
    print("[INFO] Loading dataset files...")
    df_meta = pd.read_csv(METADATA_CSV)
    
    with open(EMBEDDINGS_JSON, 'r') as f:
        embeddings_data = json.load(f)
        
    # Map embeddings to IDs
    emb_dict = {item["conversation_id"]: item["embedding"] for item in embeddings_data}
    
    # 2. Prepare feature vectors (X) and labels (y)
    X_list = []
    y_list = []
    
    for idx, row in df_meta.iterrows():
        conv_id = row["conversation_id"]
        if conv_id in emb_dict:
            X_list.append(emb_dict[conv_id])
            y_list.append(row["conversion_outcome"])
            
    X = np.array(X_list)  # Shape: (N, 3072)
    y = np.array(y_list)  # Shape: (N,)
    
    num_samples, num_features = X.shape
    print(f"[SUCCESS] Loaded {num_samples} samples with {num_features}-dimensional embeddings.")
    
    # 3. Train-Test Split
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * (1 - test_split))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"[INFO] Train size: {len(X_train)} samples | Test size: {len(X_test)} samples")
    
    # 4. Initialize weights and bias
    weights = np.random.normal(0, 0.01, num_features)
    bias = 0.0
    
    print("[INFO] Starting gradient descent...")
    
    # 5. Training loop
    for epoch in range(1, epochs + 1):
        # Forward pass (train)
        z_train = np.dot(X_train, weights) + bias
        a_train = sigmoid(z_train)
        
        # Loss calculation (Binary Cross Entropy)
        # Avoid log(0) using epsilon clipping
        a_train_clipped = np.clip(a_train, 1e-15, 1.0 - 1e-15)
        loss_train = -np.mean(y_train * np.log(a_train_clipped) + (1 - y_train) * np.log(1 - a_train_clipped))
        
        # Backward pass (Gradients)
        dw = np.dot(X_train.T, (a_train - y_train)) / len(y_train)
        db = np.mean(a_train - y_train)
        
        # Parameter Updates
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Evaluation on test set
        if epoch % 10 == 0 or epoch == 1:
            z_test = np.dot(X_test, weights) + bias
            a_test = sigmoid(z_test)
            y_pred_test = (a_test >= 0.5).astype(int)
            accuracy_test = np.mean(y_pred_test == y_test)
            
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {loss_train:.5f} | Test Accuracy: {accuracy_test:.2%}")
            
    # Final Metrics
    z_test = np.dot(X_test, weights) + bias
    a_test = sigmoid(z_test)
    y_pred_test = (a_test >= 0.5).astype(int)
    
    tp = np.sum((y_pred_test == 1) & (y_test == 1))
    fp = np.sum((y_pred_test == 1) & (y_test == 0))
    fn = np.sum((y_pred_test == 0) & (y_test == 1))
    tn = np.sum((y_pred_test == 0) & (y_test == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE METRICS (TEST SET)")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1-Score  : {f1:.4f}")
    print("=" * 60)
    
    # 6. Save weights to JSON
    model_data = {
        "model_type": "LogisticRegressionClassifier",
        "num_features": num_features,
        "weights": weights.tolist(),
        "bias": float(bias),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
    }
    
    with open(MODEL_OUTPUT_JSON, 'w') as f:
        json.dump(model_data, f)
    print(f"[SUCCESS] Trained model weights saved to: {MODEL_OUTPUT_JSON}")

if __name__ == "__main__":
    train_model(epochs=120, learning_rate=0.25)

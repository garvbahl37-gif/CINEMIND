import mlflow
import mlflow.sklearn
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock training data to simulate the process
def load_data():
    logger.info("Loading training data...")
    time.sleep(1) # Simulate I/O
    return np.random.rand(100, 5), np.random.randint(0, 2, 100)

def train_model():
    """
    Simulates a training pipeline that logs to MLflow.
    In a real scenario, this would retrain the Sentence-BERT model or FAISS index.
    """
    # Connect to MLflow (assuming it's running in Docker or localhost)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MovieRec_Embeddings_v1")

    with mlflow.start_run():
        logger.info("Starting training run...")
        
        # 1. Log Hyperparameters
        params = {
            "embedding_dim": 384,
            "min_vote_count": 1000,
            "model_type": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "epochs": 5
        }
        mlflow.log_params(params)
        logger.info(f"Logged params: {params}")

        # 2. Simulate Training Loop
        X, y = load_data()
        
        for epoch in range(params["epochs"]):
            # Simulate changing metrics
            loss = 0.5 * (0.9 ** epoch) + np.random.normal(0, 0.01)
            accuracy = 0.8 + (0.02 * epoch) + np.random.normal(0, 0.005)
            
            # Log metrics per step
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            logger.info(f"Epoch {epoch+1}/{params['epochs']}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
            time.sleep(0.5)

        # 3. Register Model (Simulated)
        # In reality, you'd save the FAISS index or PyTorch model here
        logger.info("Saving model artifacts...")
        with open("production.index", "w") as f:
            f.write("simulation_index_data")
        
        mlflow.log_artifact("production.index")
        
        logger.info("✅ Training complete! View results at http://localhost:5000")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.warning("Ensure MLflow is running (docker-compose up -d mlflow)")

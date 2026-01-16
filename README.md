# 🎬 CineMind - Advanced AI Recommender System (Production Grade)

> **A scalable, event-driven Movie Recommendation Engine powered by FastAPI, React, Kafka, Redis, and MLOps.**

![Production Status](https://img.shields.io/badge/Status-Production%20Ready-green?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20|%20React%20|%20Docker%20|%20Kafka%20|%20Redis-blue?style=for-the-badge)

## 🏗️ System Architecture

This project simulates a **real-world "FAANG-scale" architecture** designed to handle high traffic, real-time events, and continuous model improvement.

```mermaid
graph TD
    User[User] -->|Browser| Frontend[React Frontend (Container)]
    Frontend -->|API Reqs| Backend[FastAPI Backend (Container)]
    
    subgraph "Data & Performance Layer"
        Backend -->|Cache Hit| Redis[Redis Cache (Hot Data)]
        Backend -->|Cache Miss| FAISS[FAISS Index (Vector Search)]
    end
    
    subgraph "Event Streaming (Real-Time)"
        Backend -->|User Clicks/Search| Kafka[Apache Kafka]
        Kafka -->|Topic: user_events| StreamProcess[spark-streaming (Future)]
    end
    
    subgraph "MLOps (Model Lifecycle)"
        FAISS -->|Model Metrics| MLflow[MLflow Server]
        MLflow -->|Experiment Tracking| Dashboard[MLflow UI]
    end
```

## 🚀 Key Features

*   **⚡ Hybrid Search Engine**: Combines **Sentence-BERT Embeddings (Vector Search)** with traditional keyword matching for 100% recall.
*   **🏎️ Lightning Performance**: **Redis Caching** stores "Top 50" and "TV Shows" results, reducing API latency from ~200ms to **<5ms**.
*   **📡 Real-Time Event Streaming**: **Apache Kafka** captures user interactions (searches) instantly for future training.
*   **🧪 MLOps Integrated**: **MLflow** tracks model experiments and accuracy, ensuring reproducible AI.
*   **🐳 Fully Dockerized**: The entire stack launches with a single command.

## 🛠️ Prerequisites

To run the full stack, you need:

1.  **Docker Desktop** (Essential): Run all services without installing Python/Node manually.
2.  **Git**: To clone the repo.

## 🏃‍♂️ How to Run (Production Mode)

1.  **Clone & Enter Directory**:
    ```bash
    git clone <repo-url>
    cd Advance-Recommender-System
    ```

2.  **Launch the Stack**:
    ```bash
    docker-compose up --build
    ```
    *Wait for Docker to download images and build the containers.*

3.  **Access Services**:
    *   🍿 **Web App**: [http://localhost:5173](http://localhost:5173)
    *   🧠 **Backend API**: [http://localhost:7860/docs](http://localhost:7860/docs)
    *   📊 **MLFlow UI**: [http://localhost:5000](http://localhost:5000)

## 📂 Project Structure

```bash
├── 📁 deployment/          # Docker & Cloud Configs (App Logic)
│   ├── Dockerfile          # Backend Container
│   ├── app.py              # Main FastAPI Application
│   ├── kafka_utils.py      # Kafka Producer
│   └── requirements.txt    # Dependencies
├── 📁 frontend-app/        # React Application
│   ├── src/                # UI Components
│   └── Dockerfile          # Frontend Container
├── 📁 mlops/               # AI Engineering
│   └── train.py            # Training Pipeline & MLflow
├── 📁 ml-32m/              # Dataset (if local)
├── docker-compose.yml      # Orchestration Script
└── README.md               # Documentation
```

## 🧠 Advanced: Research Pipeline (Legacy)

For data scientists wanting to retrain the underlying models from scratch:

```bash
# Install local dependencies
pip install -r requirements.txt

# Run the training pipeline
python run.py train --epochs 50
```

See `run.py` for full CLI options regarding the Two-Tower Architecture and FAISS Indexing.

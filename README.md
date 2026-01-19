# CINEMIND - Advanced Recommender System

CINEMIND is a production-grade movie recommendation engine designed to simulate high-scale, modern application architectures. It leverages a microservices approach, integrating a FastAPI backend, a React-based frontend, and robust data engineering pipelines powered by Apache Kafka, Redis, and MLflow.

## Project Overview

This system is built to provide personalized movie recommendations using a Two-Tower Neural Network architecture for candidate generation and FAISS for efficient vector similarity search. It features a modern, responsive user interface and a backend capable of handling real-time search, filtering, and user interactions.

## Technology Stack

### Core Application
- **Frontend**: React (TypeScript), TailwindCSS, Framer Motion, Vite
- **Backend**: Python, FastAPI, Uvicorn
- **Containerization**: Docker, Docker Compose

### Data & Machine Learning
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Machine Learning**: PyTorch, Scikit-learn
- **Embeddings**: Sentence-BERT (SBERT) for semantic text search
- **Experiment Tracking**: MLflow
- **Data Streaming**: Apache Kafka (User event tracking)
- **Caching**: Redis (High-performance API caching)

## Key Features

1.  **Hybrid Search Engine**: Combines semantic vector search with traditional keyword matching (BM25 logic) to provide accurate search results for titles, genres, and metadata.
2.  **Personalized Recommendations**: Utilizes user and item embeddings to suggest movies similar to user preferences or currently viewed items ("Also Liked" functionality).
3.  **Real-Time Performance**: Implements Redis caching for frequently accessed data (Top 50, TV Shows), reducing API response times to single-digit milliseconds.
4.  **Intelligent Intent Parsing**: Integrates with Large Language Models (LLM) to parse complex user queries (e.g., "90s action movies") into structured search filters.
5.  **Reactive UI/UX**: Features a "Warm Luxe" aesthetic with glassmorphism effects, dynamic animations, and a fully responsive design.
6.  **Event-Driven Architecture**: Captures user interactions such as searches and clicks via Kafka topics for future model retraining and analytics.

## Project Structure

```
CINEMIND/
├── deployment/                 # Production-ready backend configurations
│   ├── app.py                  # Main FastAPI application entry point
│   ├── llm_engine.py           # LLM integration for intent parsing
│   ├── movies.json             # Core metadata dataset
│   └── Dockerfile              # Backend container definition
│
├── frontend-app/               # React Frontend Application
│   ├── src/                    # Source code
│   │   ├── components/         # Reusable UI components
│   │   ├── App.tsx             # Main application logic
│   │   └── config.ts           # API configuration
│   └── Dockerfile              # Frontend container definition
│
├── mlops/                      # Machine Learning Operations
│   └── train.py                # Training pipelines and MLflow integration
│
├── candidate_generation/       # Recommendation Algorithms
│   └── two_tower/              # Neural network architecture definition
│
├── data/                       # Data processing scripts and storage
└── docker-compose.yml          # Container orchestration configuration
```

## Installation and Setup

### Prerequisites
- Docker Desktop
- Git
- Node.js (Optional, for local frontend development)
- Python 3.10+ (Optional, for local backend development)

### Running with Docker (Recommended)
The entire application stack can be launched using Docker Compose.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/garvbahl37-gif/CINEMIND.git
    cd CINEMIND
    ```

2.  **Start the services:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the application:**
    - Frontend: http://localhost:5173
    - Backend API Docs: http://localhost:7860/docs

### Local Development Setup

**Backend:**
1.  Navigate to the project root.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Start the server:
    ```bash
    cd deployment
    python -m uvicorn app:app --host 0.0.0.0 --port 8003
    ```

**Frontend:**
1.  Navigate to `frontend-app`.
2.  Install dependencies: `npm install`
3.  Start the development server: `npm run dev`

## Deployment

The application is designed for cloud deployment:
- **Frontend**: Deployed on Vercel (https://cinemind-theta.vercel.app/)
- **Backend**: Hosted on Hugging Face Spaces (https://bharatverse11-movie-recommender-system.hf.space)

## License
This project is licensed under the MIT License.

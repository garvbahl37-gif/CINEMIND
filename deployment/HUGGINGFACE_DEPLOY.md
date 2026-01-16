# Hugging Face Spaces Deployment Guide
## MovieRec AI - FAANG-Grade Recommender System

---

## 🎯 Space Settings

When creating your Hugging Face Space, use these settings:

| Setting | Value |
|---------|-------|
| **Space name** | `movie-recommender-api` |
| **SDK** | Docker |
| **Hardware** | CPU Basic (Free) or CPU Upgrade |
| **Visibility** | Public |

---

## 📁 Files to Upload

Upload these files to your Space repository:

### Required Files (Core API)

```
├── app.py                    # Main FastAPI application (renamed from serving/api/main.py)
├── recommendation_service.py # Service logic
├── requirements.txt          # Dependencies
├── Dockerfile                # Container configuration
└── README.md                 # Space description (optional)
```

### If Using Pre-trained Embeddings (Full System)

```
├── app.py
├── recommendation_service.py
├── requirements.txt
├── Dockerfile
├── embeddings/
│   ├── user_embeddings.npy   # Pre-computed user embeddings
│   └── item_embeddings.npy   # Pre-computed item embeddings
├── indices/
│   └── production.index      # FAISS index
├── data/processed/
│   ├── metadata.json         # Dataset metadata
│   └── encoders.npz          # ID encoders
└── candidate_generation/
    └── ann/
        └── faiss_index.py    # FAISS manager
```

---

## 📝 Files Content

### 1. `Dockerfile` (Required)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 2. `requirements.txt` (Simplified for HF)

```
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
faiss-cpu>=1.7.4
```

### 3. `app.py` (Simplified Demo API)

The simplified demo API is provided in the `deployment/app.py` file.

---

## 🚀 Step-by-Step Deployment

### Option A: Demo Mode (No Training Required)

1. **Create Space**: Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space
2. **Settings**:
   - SDK: Docker
   - Hardware: CPU Basic (Free)
3. **Upload Files**:
   - `deployment/Dockerfile`
   - `deployment/app.py`
   - `deployment/requirements.txt`
4. **Wait for Build**: ~2-3 minutes
5. **Access API**: `https://huggingface.co/spaces/YOUR_USERNAME/movie-recommender-api`

### Option B: Full System (With Trained Model)

1. Run training locally:
   ```bash
   python run.py full --sample 100000 --epochs 5
   ```
2. Upload additional files:
   - `embeddings/item_embeddings.npy`
   - `embeddings/user_embeddings.npy`
   - `indices/production.index`
   - `data/processed/metadata.json`
3. Use the full `app.py` from `serving/api/main.py`

---

## 🔗 API Endpoints

Once deployed, your API will be available at:

```
https://YOUR_USERNAME-movie-recommender-api.hf.space
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/recommend/{user_id}` | GET | Get recommendations |
| `/similar/{item_id}` | GET | Get similar items |

### Example Usage

```bash
# Health check
curl https://YOUR_USERNAME-movie-recommender-api.hf.space/health

# Get recommendations
curl https://YOUR_USERNAME-movie-recommender-api.hf.space/recommend/1?k=10

# Get similar items
curl https://YOUR_USERNAME-movie-recommender-api.hf.space/similar/100?k=5
```

---

## ⚙️ Environment Variables (Optional)

You can set these in Space Settings → Variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | `true` | Use demo data if no embeddings |
| `NUM_ITEMS` | `87000` | Number of items |
| `EMBEDDING_DIM` | `64` | Embedding dimension |

---

## 🎨 Connecting React Frontend

Update the React frontend's API URL:

```javascript
// In frontend-app/src/App.jsx
const API_BASE = 'https://YOUR_USERNAME-movie-recommender-api.hf.space'
```

Then deploy the frontend to:
- **Vercel**: `vercel deploy`
- **Netlify**: Drag & drop `dist` folder
- **GitHub Pages**: Push to `gh-pages` branch

---

## 📊 Hardware Recommendations

| Scale | Hardware | Cost |
|-------|----------|------|
| Demo/Testing | CPU Basic | Free |
| Light Production | CPU Upgrade | $0.03/hr |
| Heavy Traffic | A10G Small | $1.05/hr |

For FAISS with 87K items, CPU Basic is sufficient (~100ms latency).

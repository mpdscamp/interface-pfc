# PFC WebApp - Network Traffic ML Platform

A minimal viable product (MVP) for fine-tuning and running inference on network traffic datasets using DistilBERT models.

## Features

- **Dataset Management**: Upload CSV network traffic datasets
- **Model Training**: Start fine-tuning jobs for `distilbert-base-uncased` models
- **Inference**: Run inference with trained model checkpoints
- **Results Visualization**: View accuracy metrics and confusion matrices
- **Job Monitoring**: Real-time status updates for all training and inference jobs

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pfc-webapp
   ```

2. **Prepare your datasets**
   Place your CSV files in the `datasets/` directory:
   ```bash
   mkdir -p datasets
   cp your-network-traffic-data.csv datasets/
   ```

3. **Start the application**
   ```bash
   docker compose up --build -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Project Structure

```
/pfc-webapp/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # API routes
│   │   ├── models.py       # Database models
│   │   ├── schemas.py      # Pydantic schemas
│   │   ├── tasks.py        # Background tasks
│   │   └── database.py     # SQLite setup
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   └── App.tsx         # Main app component
│   ├── Dockerfile
│   └── package.json
├── datasets/               # CSV datasets (volume)
├── models_output/          # Model checkpoints and results
│   ├── checkpoints/
│   └── inference_results/
├── pfc_webapp.db          # SQLite database
└── docker-compose.yml
```

## Usage Guide

### 1. Upload Datasets
Navigate to the Training page and use the upload feature to add CSV files to the system.

### 2. Train a Model
- Go to the Training page
- Select a dataset from the dropdown
- Enter a model name (default: `distilbert-base-uncased`)
- Click "Start Training"
- Monitor progress in the Job Status table

### 3. Run Inference
- Go to the Inference page
- Select a trained model checkpoint
- Choose a dataset for inference
- Click "Run Inference"
- View results when the job completes

### 4. View Results
- Click "View Results" in the Job Status table for completed inference jobs
- Review accuracy metrics and confusion matrix

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/api/jobs/train` | Start a training job |
| POST   | `/api/jobs/infer` | Start an inference job |
| GET    | `/api/jobs` | List all jobs |
| GET    | `/api/jobs/{job_id}` | Get job status |
| GET    | `/api/datasets` | List available datasets |
| POST   | `/api/datasets/upload` | Upload a new dataset |
| GET    | `/api/models` | List trained models |
| GET    | `/api/results/{job_id}` | Get inference results |

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## Troubleshooting

### Container Issues
- Check logs: `docker compose logs -f [backend|frontend]`
- Restart services: `docker compose restart`
- Rebuild containers: `docker compose up --build`

### Database Issues
- Database is stored in `pfc_webapp.db`
- To reset: delete the file and restart backend

### Port Conflicts
- Frontend runs on port 3000
- Backend runs on port 8000
- Change ports in `docker-compose.yml` if needed

## Notes

- This is an MVP with simulated training/inference (using sleep)
- Real ML implementation would replace the mock functions in `tasks.py`
- SQLite database is used for simplicity (single file, no setup)
- All data is persisted via Docker volumes
from uuid import uuid4
from typing import List, Dict

import os, json, threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import SessionLocal, engine
from .models import Base, Job
from .schemas import JobCreateTrain, JobCreateInfer, JobStatus, InferenceResult
from .tasks import run_job
from .llm_pipeline import save_checkpoint_llm, stop_training_llm

# --------------------------------------------------------------------------- #
#                            FASTAPI  BOOTSTRAP                               #
# --------------------------------------------------------------------------- #
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS_DIR = "/datasets"
CHECKPOINT_DIR = "models_output/checkpoints"
LLM_DIR = "models_output/llm_checkpoints"


# --------------------------------------------------------------------------- #
#                           DB helper (yield style)                           #
# --------------------------------------------------------------------------- #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
#                               JOB ENDPOINTS                                 #
# --------------------------------------------------------------------------- #
@app.post("/api/jobs/train", response_model=JobStatus)
# def create_train_job(payload: JobCreateTrain, background_tasks: BackgroundTasks):
def create_train_job(payload: JobCreateTrain):
    """
    payload.kind  →  "tabular" | "llm"
    """
    job_id = str(uuid4())
    job = Job(
        id=job_id,
        kind=f"{payload.kind}_train",
        model_name=payload.model_name,
        dataset_filename=payload.dataset_filename,
        status="QUEUED",
        progress=0,
    )
    db: Session = next(get_db())
    db.add(job); db.commit()

    # background_tasks.add_task(run_job, job_id)
    t = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    t.start()
    return JobStatus.model_validate(job)


@app.post("/api/jobs/infer", response_model=JobStatus)
# def create_infer_job(payload: JobCreateInfer, background_tasks: BackgroundTasks):
def create_infer_job(payload: JobCreateInfer):
    """
    payload.checkpoint is:
        – *.joblib                   (tabular)
        – <folder name under LLM_DIR>  (llm)
    """
    job_id = str(uuid4())
    kind = f"{payload.kind}_infer"
    job_kwargs = dict(
        id=job_id,
        kind=kind,
        model_name="N/A",
        dataset_filename=payload.dataset_filename,
        status="QUEUED",
        progress=0,
    )

    if payload.kind == "tabular":
        job_kwargs["checkpoint_filename"] = payload.checkpoint
    else:
        job_kwargs["checkpoint_dir"] = payload.checkpoint

    db: Session = next(get_db())
    db.add(Job(**job_kwargs)); db.commit()

    # background_tasks.add_task(run_job, job_id)
    t = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    t.start()
    return JobStatus.from_orm(Job(**job_kwargs))


@app.get("/api/jobs", response_model=List[JobStatus])
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.submitted_at.desc()).all()
    return [JobStatus.model_validate(j) for j in jobs]


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus.from_orm(job)


@app.post("/api/jobs/{job_id}/checkpoint/save")
def api_save_checkpoint(job_id: str):
    db = next(get_db())
    job = db.query(Job).get(job_id)
    if not job or job.kind != "llm_train" or job.status != "RUNNING":
        raise HTTPException(400, "No running LLM training job with this ID")
    try:
        save_checkpoint_llm(job_id)
        return {"message": "Checkpoint requested"}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/jobs/{job_id}/checkpoint/stop")
def api_stop_training(job_id: str):
    db = next(get_db())
    job = db.query(Job).get(job_id)
    if not job or job.kind != "llm_train" or job.status != "RUNNING":
        raise HTTPException(400, "No running LLM training job with this ID")
    try:
        stop_training_llm(job_id)
        return {"message": "Stop requested"}
    except RuntimeError as e:
        raise HTTPException(400, str(e))

# --------------------------------------------------------------------------- #
#                        DATASET &  MODEL ENDPOINTS                           #
# --------------------------------------------------------------------------- #
@app.get("/api/datasets", response_model=List[Dict])
def list_datasets():
    return [
        {
            "filename": fn,
            "size_mb": round(os.path.getsize(os.path.join(DATASETS_DIR, fn)) / 1_048_576, 2),
        }
        for fn in os.listdir(DATASETS_DIR)
        if fn.lower().endswith(".csv")
    ]


@app.post("/api/datasets/upload", response_model=Dict)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")
    dest = os.path.join(DATASETS_DIR, file.filename)
    with open(dest, "wb") as out:
        out.write(await file.read())
    return {"filename": file.filename, "status": "uploaded"}


@app.get("/api/models", response_model=List[Dict])
def list_models():
    """
    Combines both model families so the frontend can show one dropdown.
    Returns: [{filename/display_name/kind}, …]
    """
    models = []

    # tabular
    for fn in sorted(os.listdir(CHECKPOINT_DIR)):
        if fn.endswith(".joblib"):
            ts = fn.split("__")[-1].split(".")[0]
            models.append(
                {
                    "filename": fn,
                    "display_name": f"{fn.split('__')[0]} @ {ts}",
                    "kind": "tabular",
                }
            )

    # llm
    if os.path.isdir(LLM_DIR):
        for d in sorted(os.listdir(LLM_DIR)):
            if os.path.isdir(os.path.join(LLM_DIR, d)):
                models.append(
                    {
                        "filename": d,
                        "display_name": f"LLM {d}",
                        "kind": "llm",
                    }
                )
    return models


@app.get("/api/results/{job_id}", response_model=InferenceResult)
def get_results(job_id: str):
    db: Session = next(get_db())
    job = db.query(Job).get(job_id)
    if not job or not job.result_path:
        raise HTTPException(404, "Results not found")
    with open(job.result_path) as fp:
        data = json.load(fp)
    return InferenceResult(**data)

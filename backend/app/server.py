# server.py
import os
import uuid
import datetime
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.cloud import storage

# ------------------------- Paths & Config -------------------------

# Resolve directories based on this file's location:
#   .../Imagicle/backend/app/server.py
# so APP_DIR = .../backend/app
#    BACKEND_DIR = .../backend
#    REPO_ROOT = .../Imagicle
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

def _sanitize_bucket(name: str) -> str:
    # Accept "gs://bucket" or "bucket" env values; normalize to "bucket"
    return name.replace("gs://", "").strip().strip("/")

BUCKET = _sanitize_bucket(os.getenv("POINTCLOUD_BUCKET", "imagicle-473400-pointclouds-dev"))

# Resolve Point-E script:
# - If POINT_E_SCRIPT env is set and relative, treat it as relative to REPO_ROOT
# - Otherwise default to backend/vendor/point-e/... under this repo
_env_point_e = os.getenv("POINT_E_SCRIPT")
if _env_point_e:
    _p = Path(_env_point_e)
    POINT_E_SCRIPT = _p if _p.is_absolute() else (REPO_ROOT / _p)
else:
    POINT_E_SCRIPT = BACKEND_DIR / "vendor/point-e/point_e/evals/scripts/generate.py"

# Where to store optional local copies of generated PLYs for dev viewing
_env_artifacts = os.getenv("ARTIFACTS_DIR", str(BACKEND_DIR / "artifacts"))
ARTIFACTS_DIR = Path(_env_artifacts)
if not ARTIFACTS_DIR.is_absolute():
    ARTIFACTS_DIR = BACKEND_DIR / ARTIFACTS_DIR
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Frontend origins allowed to call THIS FastAPI (separate from GCS bucket CORS)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173"
).split(",") if o.strip()]

# ------------------------- App & Middleware -------------------------

app = FastAPI(title="imagicle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve local dev artifacts (optional)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

# ------------------------- Models -------------------------

class GenerateReq(BaseModel):
    prompt: str
    user_id: str | None = None
    # Optional tuning knobs you may expose later:
    guidance: float | None = None
    seed: int | None = None
    no_upsample: bool | None = None

# ------------------------- Routes -------------------------

@app.get("/healthz")
def health():
    return {
        "ok": True,
        "service": "imagicle",
        "bucket": BUCKET,
        "point_e_script": str(POINT_E_SCRIPT),
        "artifacts_dir": str(ARTIFACTS_DIR),
    }

@app.post("/api/generate")
def generate_pointcloud(req: GenerateReq):
    # 0) Sanity checks
    if not POINT_E_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Point-E script not found at {POINT_E_SCRIPT}")

    # 1) Unique job id + local tmp output
    job_id = str(uuid.uuid4())[:8]
    user = (req.user_id or "anon").replace("/", "_")
    local_out = Path(f"/tmp/pointcloud_{job_id}.ply")

    # 2) Build Point-E command
    cmd = ["python3", str(POINT_E_SCRIPT), "--prompt", req.prompt, "--out", str(local_out)]
    if req.guidance is not None:
        cmd += ["--guidance", str(req.guidance)]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    if req.no_upsample:
        cmd += ["--no_upsample"]

    # 3) Run Point-E synchronously (capture logs for debugging)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not local_out.exists():
        msg = (proc.stderr or proc.stdout or "Point-E failed")[:4000]
        raise HTTPException(status_code=500, detail=f"Point-E error:\n{msg}")

    # 4) Upload to GCS
    object_path = f"pointclouds/{user}/{job_id}/output.ply"
    client = storage.Client()  # On GCE, uses the VM's service account
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(object_path)
    blob.content_type = "application/octet-stream"
    blob.upload_from_filename(str(local_out))

    # 5) Optional: keep a local copy for quick dev viewing
    try:
        (ARTIFACTS_DIR / f"{job_id}.ply").write_bytes(local_out.read_bytes())
    except Exception:
        pass  # non-fatal

    # 6) Return a short-lived signed URL
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="GET",
        response_disposition='inline; filename="output.ply"',
    )
    return {
        "job_id": job_id,
        "gcs_uri": f"gs://{BUCKET}/{object_path}",
        "url": url
    }

@app.get("/api/pointcloud/url")
def sign_existing(object_path: str = Query(..., description="e.g. pointclouds/anon/<job>/output.ply")):
    """Sign an existing object path and return a temporary GET URL."""
    client = storage.Client()
    blob = client.bucket(BUCKET).blob(object_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Object not found")
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="GET",
        response_disposition='inline; filename="output.ply"',
    )
    return {"url": url}

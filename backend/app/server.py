# server.py
import os
import re
import uuid
import datetime
import subprocess
from pathlib import Path
import sys

from fastapi.responses import RedirectResponse, Response
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

OUTPUT_DIR = BACKEND_DIR / "data" / "outputs" / "pointclouds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
@app.get("/", include_in_schema=False)
def index():
    # Visiting http://localhost:8000/ sends you to the Swagger docs
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Silence the 404 spam for /favicon.ico during dev
    return Response(status_code=204)

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
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid.uuid4())[:8]
    user = (req.user_id or "anon").replace("/", "_")
    local_out = OUTPUT_DIR / f"{job_id}.ply"

    # Run Point-E via your current interpreter; set cwd so relative paths resolve
    cmd = [
        sys.executable, "-m", "point_e.evals.scripts.generate",
        "--prompt", req.prompt,
        "--out", str(local_out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # If our expected file isn't there, parse the actual saved path from stdout
    if not local_out.exists():
        m = re.search(r"Saved:\s*(.+\.ply)", proc.stdout or "")
        if m:
            candidate = Path(m.group(1))
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            if candidate.exists():
                local_out = candidate

    if proc.returncode != 0 or not local_out.exists():
        msg = (proc.stderr or proc.stdout or "Point-E failed")[:4000]
        raise HTTPException(status_code=500, detail=f"Point-E error:\n{msg}")

    # Upload to GCS (same as before)
    object_path = f"pointclouds/{user}/{job_id}/output.ply"
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(object_path)
    blob.content_type = "application/octet-stream"
    blob.upload_from_filename(str(local_out))

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="GET",
        response_disposition='inline; filename="output.ply"',
    )
    return {"job_id": job_id, "gcs_uri": f"gs://{BUCKET}/{object_path}", "url": url}
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

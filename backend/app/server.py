from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="imagicle API")

# serve cached artifacts (PLY/GLB) directly
app.mount("/artifacts", StaticFiles(directory="backend/artifacts"), name="artifacts")

@app.get("/healthz")
def health():
    return {"ok": True, "service": "imagicle"}

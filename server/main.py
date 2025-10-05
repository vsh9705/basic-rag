# server/main.py
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from logger import logger

from modules.policy_store import build_or_update_policy_store, load_policy_store, clear_policy_store, PERSIST_POLICY_DIR
from modules.moderation import moderate_file_against_policy

app = FastAPI(title="Content Moderation (Policy-based)")

# allow frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def catch_exception_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/test")
async def test():
    return {"message": "Moderation server is up and running!"}

@app.post("/upload_policy/")
async def upload_policy(files: List[UploadFile] = File(...)):
    """
    Upload one or more policy PDFs. These are persisted into the policy_store (Chroma).
    """
    try:
        logger.info(f"Received {len(files)} policy files for upload")
        store = build_or_update_policy_store(files)
        return {"message": "Policy files processed and stored", "policy_store_path": PERSIST_POLICY_DIR}
    except Exception as e:
        logger.exception("Error during policy upload")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/moderate/")
async def moderate(current_file: UploadFile = File(...)):
    """
    Upload a single current file (PDF) to be checked against the persisted policy store.
    Returns a moderation verdict, any violations and sources.
    """
    try:
        # 1) Ensure policy store exists
        try:
            policy_store = load_policy_store()
        except FileNotFoundError as fe:
            return JSONResponse(status_code=400, content={"error": "Policy store empty. Upload policy PDFs first."})

        # 2) Run moderation
        result = moderate_file_against_policy(policy_store, current_file, k=3)
        return result
    except Exception as e:
        logger.exception("Error during moderation")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/clear_policy/")
async def clear_policy():
    """
    Clear persistent policy store and uploaded policies.
    """
    try:
        clear_policy_store()
        return {"message": "Policy store cleared"}
    except Exception as e:
        logger.exception("Error clearing policy store")
        return JSONResponse(status_code=500, content={"error": str(e)})

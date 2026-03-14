from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.routes.ingest import ingest_log
from api.security import require_api_key

router = APIRouter(tags=["analyze"])


@router.post("/analyze")
def analyze_log(raw: dict, request: Request, _auth: None = Depends(require_api_key)):
    return ingest_log(raw, request)

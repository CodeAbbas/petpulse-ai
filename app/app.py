from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────

MODEL_PATH = "models/yolo26n.pt"  # ONNX export added Day 9
CONFIDENCE_THRESHOLD = 0.25
TARGET_CLASSES = {"dog", "cat"}

# Module-level model handle, loaded once at startup.
model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the YOLO26n model once on startup; release on shutdown."""
    global model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 — surface load failure clearly
        raise RuntimeError(f"Failed to load YOLO26n model: {exc}") from exc
    yield
    model = None


app = FastAPI(
    title="PetPulse Edge AI",
    description="YOLO26n perception service for separation-anxiety detection",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Response schemas ────────────────────────────────────────────

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    centre: list[float]  # [cx, cy]


class DetectionResponse(BaseModel):
    detections: list[Detection]
    frame_width: int
    frame_height: int
    # Populated by Zone-Based Logic (Day 9):
    anxiety_event: bool = False
    event_type: str | None = None
    confidence_score: float | None = None


# ─── Helpers ─────────────────────────────────────────────────────

def _decode_image(raw: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR ndarray."""
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Unable to decode image payload.")
    return frame


def _run_inference(frame: np.ndarray) -> list[Detection]:
    """Run YOLO26n inference and return filtered pet detections."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections: list[Detection] = []

    for result in results:
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls)]
            if label not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf),
                    bbox=[x1, y1, x2, y2],
                    centre=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                )
            )

    return detections


# ─── Routes ──────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect/separation-anxiety", response_model=DetectionResponse)
async def detect_separation_anxiety(
    frame: UploadFile = File(...),
) -> DetectionResponse:
    """Accept an image frame, run detection, and (Day 9) apply Zone-Based Logic."""
    raw = await frame.read()
    image = _decode_image(raw)
    height, width = image.shape[:2]

    detections = _run_inference(image)

    # ─────────────────────────────────────────────────────────────
    # ZONE-BASED LOGIC PLACEHOLDER (Day 9)
    #
    # Tomorrow's implementation:
    #   1. Test each detection centre against the configured Door Zone
    #      polygon (cv2.pointPolygonTest).
    #   2. Accumulate Time-in-Zone per pet across frames.
    #   3. Detect rapid entry/exit (>=3 crossings in 60s) → pacing.
    #   4. On threshold breach: set anxiety_event=True, event_type,
    #      confidence_score, generate a UUID, and dispatch an async
    #      webhook to POST {API}/behavioral-events.
    # ─────────────────────────────────────────────────────────────

    return DetectionResponse(
        detections=detections,
        frame_width=width,
        frame_height=height,
        anxiety_event=False,
        event_type=None,
        confidence_score=None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
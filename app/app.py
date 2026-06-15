from __future__ import annotations

import os
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt")
CONFIDENCE_THRESHOLD = 0.25
TARGET_CLASSES = {"dog", "cat"}

LARAVEL_WEBHOOK_URL = os.getenv(
    "LARAVEL_WEBHOOK_URL", "http://127.0.0.1:8000/api/v1/behavioral-events"
)
WEBHOOK_SECRET = os.getenv("PETPULSE_WEBHOOK_SECRET", "change-me-in-env")

# Door Zone polygon (normalised pixel coords for a 640x480 frame).
# Replaced with per-camera calibration in production.
DOOR_ZONE_POLYGON = np.array(
    [[420, 80], [620, 80], [620, 460], [420, 460]], dtype=np.int32
)

# Pacing heuristic: N zone crossings within the rolling window (seconds).
PACING_CROSSINGS = 3
PACING_WINDOW_SECONDS = 60.0

# Prolonged-waiting heuristic: continuous time-in-zone threshold (seconds).
PROLONGED_WAIT_SECONDS = 300.0  # 5 minutes

model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load YOLO26n model: {exc}") from exc
    yield
    model = None


app = FastAPI(
    title="PetPulse Edge AI",
    description="YOLO26n perception + Zone-Based Logic",
    version="0.2.0",
    lifespan=lifespan,
)


# ─── In-memory state tracker ─────────────────────────────────────

class ZoneStateTracker:
    """Lightweight per-pet state for pacing and prolonged-waiting heuristics.

    All state is in-memory and ephemeral — acceptable for a single-camera
    prototype. A production multi-camera deployment would externalise this
    to Redis with TTL eviction.
    """

    def __init__(self) -> None:
        # Rolling window of recent crossing timestamps per pet.
        self._crossings: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=PACING_CROSSINGS * 4)
        )
        # Whether the pet was inside the zone on the previous frame.
        self._was_inside: dict[str, bool] = defaultdict(bool)
        # Timestamp the pet continuously entered the zone (None if outside).
        self._entered_at: dict[str, float | None] = defaultdict(lambda: None)
        # Last event dispatch time per (pet, event_type) for debouncing.
        self._last_event: dict[tuple[str, str], float] = {}

    def update(
        self, pet_id: str, is_inside: bool, now: float
    ) -> tuple[str, float] | None:
        """Update state; return (event_type, confidence) if an event fires."""
        was_inside = self._was_inside[pet_id]

        # Detect a crossing (transition into or out of the zone).
        if is_inside != was_inside:
            self._crossings[pet_id].append(now)
            if is_inside:
                self._entered_at[pet_id] = now
            else:
                self._entered_at[pet_id] = None

        self._was_inside[pet_id] = is_inside

        # ── Pacing: N crossings within the rolling window ──
        recent = [t for t in self._crossings[pet_id] if now - t <= PACING_WINDOW_SECONDS]
        if len(recent) >= PACING_CROSSINGS and self._should_fire(pet_id, "pacing", now):
            return ("pacing", 0.85)

        # ── Prolonged waiting: continuous time-in-zone threshold ──
        entered = self._entered_at[pet_id]
        if (
            is_inside
            and entered is not None
            and (now - entered) >= PROLONGED_WAIT_SECONDS
            and self._should_fire(pet_id, "prolonged_waiting", now)
        ):
            return ("prolonged_waiting", 0.80)

        return None

    def _should_fire(self, pet_id: str, event_type: str, now: float) -> bool:
        """Debounce: suppress repeat events of the same type within the window."""
        key = (pet_id, event_type)
        last = self._last_event.get(key)
        if last is not None and now - last < PACING_WINDOW_SECONDS:
            return False
        self._last_event[key] = now
        return True


tracker = ZoneStateTracker()


# ─── Schemas ─────────────────────────────────────────────────────

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]
    centre: list[float]
    in_door_zone: bool


class DetectionResponse(BaseModel):
    detections: list[Detection]
    frame_width: int
    frame_height: int
    anxiety_event: bool = False
    event_type: str | None = None
    confidence_score: float | None = None
    event_id: str | None = None


# ─── Helpers ─────────────────────────────────────────────────────

def _decode_image(raw: bytes) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Unable to decode image payload.")
    return frame


def _is_in_door_zone(centre: tuple[float, float]) -> bool:
    """Test whether a centre point lies inside the Door Zone polygon."""
    result = cv2.pointPolygonTest(DOOR_ZONE_POLYGON, centre, measureDist=False)
    return result >= 0


def _run_inference(frame: np.ndarray) -> list[Detection]:
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
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf),
                    bbox=[x1, y1, x2, y2],
                    centre=[cx, cy],
                    in_door_zone=_is_in_door_zone((cx, cy)),
                )
            )
    return detections


async def _dispatch_webhook(
    pet_id: str, event_id: str, event_type: str, confidence: float
) -> None:
    """Fire a non-blocking POST to the Laravel behavioural-events endpoint.

    Failures are swallowed and logged — the perception loop must never block
    or crash on a webhook error (R-03 mitigation).
    """
    severity = "critical" if event_type == "pacing" else "warning"
    payload = {
        "event_id": event_id,  # client-generated UUID → idempotency key
        "pet_id": pet_id,
        "event_type": event_type,
        "severity": severity,
        "confidence_score": round(confidence, 3),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    headers = {"X-Webhook-Secret": WEBHOOK_SECRET}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(LARAVEL_WEBHOOK_URL, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        print(f"[webhook] dispatch failed for {event_id}: {exc}")


# ─── Routes ──────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect/separation-anxiety", response_model=DetectionResponse)
async def detect_separation_anxiety(
    pet_id: str,
    frame: UploadFile = File(...),
) -> DetectionResponse:
    """Detect pets, apply Zone-Based Logic, and dispatch a webhook on event."""
    raw = await frame.read()
    image = _decode_image(raw)
    height, width = image.shape[:2]
    detections = _run_inference(image)

    now = datetime.now(timezone.utc).timestamp()
    any_in_zone = any(d.in_door_zone for d in detections)

    event = tracker.update(pet_id, any_in_zone, now)

    if event is not None:
        event_type, confidence = event
        event_id = str(uuid.uuid4())
        await _dispatch_webhook(pet_id, event_id, event_type, confidence)
        return DetectionResponse(
            detections=detections,
            frame_width=width,
            frame_height=height,
            anxiety_event=True,
            event_type=event_type,
            confidence_score=confidence,
            event_id=event_id,
        )

    return DetectionResponse(
        detections=detections, frame_width=width, frame_height=height
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
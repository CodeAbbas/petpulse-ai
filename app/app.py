from __future__ import annotations

import os
import uuid
import time
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

# AT2 Compliance: Target the exported ONNX model for CPU optimization
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo26n.onnx")
CONFIDENCE_THRESHOLD = 0.25
TARGET_CLASSES = {"dog", "cat"}

LARAVEL_WEBHOOK_URL = os.getenv(
    "LARAVEL_WEBHOOK_URL", "http://127.0.0.1:8000/api/v1/behavioral-events"
)
WEBHOOK_SECRET = os.getenv("PETPULSE_WEBHOOK_SECRET", "AR9q6eCSYbPjhdfyjadgtfe")

# Door Zone polygon (normalised pixel coords for a 640x480 frame).
DOOR_ZONE_POLYGON = np.array(
    [[420, 80], [620, 80], [620, 460], [420, 460]], dtype=np.int32
)

PACING_CROSSINGS = 3
PACING_WINDOW_SECONDS = 60.0
PROLONGED_WAIT_SECONDS = 300.0  # 5 minutes

model: YOLO | None = None

# ─── R-01 Thermal Mitigation Watchdog ────────────────────────────

class AdaptiveThermalWatchdog:
    """Monitors inference latency as a proxy for CPU thermal throttling."""
    def __init__(self, target_ms: float = 66.0):
        self.latency_history = deque(maxlen=30)
        self.target_ms = target_ms
        self.current_scale = 1.0
        self.cooldown_frames = 0
        
    def update_and_get_scale(self, inference_time_ms: float) -> float:
        self.latency_history.append(inference_time_ms)
        
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return self.current_scale
            
        if len(self.latency_history) == 30:
            avg_latency = sum(self.latency_history) / 30.0
            
            if avg_latency > self.target_ms and self.current_scale > 0.5:
                print(f"[WATCHDOG] Thermal spike ({avg_latency:.1f}ms). Downscaling res.")
                self.current_scale -= 0.25
                self.latency_history.clear()
                self.cooldown_frames = 150
                
            elif avg_latency < (self.target_ms * 0.7) and self.current_scale < 1.0:
                print(f"[WATCHDOG] CPU recovered ({avg_latency:.1f}ms). Restoring res.")
                self.current_scale += 0.25
                self.latency_history.clear()
                self.cooldown_frames = 150
                
        return self.current_scale

watchdog = AdaptiveThermalWatchdog()

@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load ONNX model: {exc}") from exc
    yield
    model = None

app = FastAPI(
    title="PetPulse Edge AI",
    description="YOLO26n ONNX perception + Zone-Based Logic",
    version="0.2.0",
    lifespan=lifespan,
)

# ─── In-memory state tracker ─────────────────────────────────────

class ZoneStateTracker:
    def __init__(self) -> None:
        self._crossings: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=PACING_CROSSINGS * 4)
        )
        self._was_inside: dict[str, bool] = defaultdict(bool)
        self._entered_at: dict[str, float | None] = defaultdict(lambda: None)
        self._last_event: dict[tuple[str, str], float] = {}

    def update(
        self, pet_id: str, is_inside: bool, now: float
    ) -> tuple[str, float] | None:
        was_inside = self._was_inside[pet_id]

        if is_inside != was_inside:
            self._crossings[pet_id].append(now)
            if is_inside:
                self._entered_at[pet_id] = now
            else:
                self._entered_at[pet_id] = None

        self._was_inside[pet_id] = is_inside

        recent = [t for t in self._crossings[pet_id] if now - t <= PACING_WINDOW_SECONDS]
        if len(recent) >= PACING_CROSSINGS and self._should_fire(pet_id, "pacing", now):
            return ("pacing", 0.85)

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
    thermal_scale: float = 1.0

# ─── Helpers ─────────────────────────────────────────────────────

def _decode_image(raw: bytes) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Unable to decode image payload.")
    return frame

def _is_in_door_zone(centre: tuple[float, float]) -> bool:
    result = cv2.pointPolygonTest(DOOR_ZONE_POLYGON, centre, measureDist=False)
    return result >= 0

def _run_inference(frame: np.ndarray, current_scale: float) -> tuple[list[Detection], float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if current_scale < 1.0:
        infer_frame = cv2.resize(frame, (0,0), fx=current_scale, fy=current_scale)
    else:
        infer_frame = frame

    inf_start = time.time()
    results = model(infer_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    inference_time = (time.time() - inf_start) * 1000

    detections: list[Detection] = []

    for result in results:
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls)]
            if label not in TARGET_CLASSES:
                continue
            
            x1, y1, x2, y2 = (float(v) / current_scale for v in box.xyxy[0])
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
    return detections, inference_time

async def _dispatch_webhook(
    pet_id: str, event_id: str, event_type: str, confidence: float
) -> None:
    severity = "critical" if event_type == "pacing" else "warning"
    payload = {
        "event_id": event_id,
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
    
    raw = await frame.read()
    image = _decode_image(raw)
    height, width = image.shape[:2]
    
    current_scale = watchdog.current_scale
    detections, inference_time = _run_inference(image, current_scale)
    
    # Update thermal watchdog with the latest inference latency
    watchdog.update_and_get_scale(inference_time)

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
            thermal_scale=current_scale
        )

    return DetectionResponse(
        detections=detections, frame_width=width, frame_height=height, thermal_scale=current_scale
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
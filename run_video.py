"""PetPulse Edge AI — Video File Runner (AT3 demonstration entrypoint).

Runs the YOLO26n perception + Zone-Based Logic pipeline against a
pre-recorded .mp4 file on CPU, simulating the edge node. Draws the
door-zone polygon, detection boxes, and live state so the behaviour is
visible on screen (and screen-recordable for the AT3 video), and fires
the real behavioural-events webhook to the Laravel API when an anxiety
event triggers.

Compliance: AT2 NFR-PERF-02 (ONNX CPU Inference) & R-01 (Thermal Mitigation).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone

import cv2
import numpy as np

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: ultralytics is not installed. Activate your venv and run:\n"
        "    pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


# ─── Configuration ───────────────────────────────────────────────────────

# AT2 Compliance: Target the exported ONNX model for CPU optimization
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo26n.onnx")
CONFIDENCE_THRESHOLD = 0.25
TARGET_CLASSES = {"dog", "cat"}

LARAVEL_WEBHOOK_URL = os.getenv(
    "LARAVEL_WEBHOOK_URL", "http://127.0.0.1:8000/api/v1/behavioral-events"
)
WEBHOOK_SECRET = os.getenv("PETPULSE_WEBHOOK_SECRET", "AR9q6eCSYbPjhdfyjadgtfe")

PACING_CROSSINGS = 3
PACING_WINDOW_SECONDS = 60.0
PROLONGED_WAIT_SECONDS = 300.0


# ─── R-01 Thermal Mitigation Watchdog ────────────────────────────────────

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
                print(f"\n[WATCHDOG] Thermal/Latency spike detected ({avg_latency:.1f}ms). Downscaling resolution to shed workload.")
                self.current_scale -= 0.25
                self.latency_history.clear()
                self.cooldown_frames = 150 
                
            elif avg_latency < (self.target_ms * 0.7) and self.current_scale < 1.0:
                print(f"\n[WATCHDOG] CPU recovered ({avg_latency:.1f}ms). Restoring resolution.")
                self.current_scale += 0.25
                self.latency_history.clear()
                self.cooldown_frames = 150
                
        return self.current_scale


# ─── Advanced Zone State Tracker (Direction Reversal Pacing) ─────────────

class ZoneStateTracker:
    """Upgraded per-pet state for spatial pacing and prolonged-waiting heuristics."""
    def __init__(self) -> None:
        self._was_inside: dict[str, bool] = defaultdict(bool)
        self._entered_at: dict[str, float | None] = defaultdict(lambda: None)
        self._last_event: dict[tuple[str, str], float] = {}
        
        # 90 frames = ~3 seconds at 30fps to analyze direction flips
        self._x_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=90)
        )

    def update(
        self, pet_id: str, is_inside: bool, now: float, cx: float | None = None
    ) -> tuple[str, float] | None:
        
        # 1. Prolonged Waiting Logic
        was_inside = self._was_inside[pet_id]
        if is_inside != was_inside:
            self._entered_at[pet_id] = now if is_inside else None
        self._was_inside[pet_id] = is_inside

        entered = self._entered_at[pet_id]
        if (
            is_inside
            and entered is not None
            and (now - entered) >= PROLONGED_WAIT_SECONDS
            and self._should_fire(pet_id, "prolonged_waiting", now)
        ):
            return ("prolonged_waiting", 0.80)

        # 2. Upgraded Spatial Pacing Logic
        if cx is not None:
            self._x_history[pet_id].append(cx)
            if self._is_pacing(self._x_history[pet_id]) and self._should_fire(pet_id, "pacing", now):
                return ("pacing", 0.85)
        else:
            # Reset pacing history if pet leaves the frame to avoid false triggers
            self._x_history[pet_id].clear()

        return None

    def _is_pacing(self, history: deque[float]) -> bool:
        if len(history) < 30: 
            return False

        history_list = list(history)
        deltas = np.diff(history_list)

        # Filter out tiny jitters (shifts less than 2 pixels)
        directions = []
        for d in deltas:
            if d > 2.0:
                directions.append(1)  # Moving right
            elif d < -2.0:
                directions.append(-1) # Moving left
            else:
                directions.append(0)  # Stationary

        sign_changes = 0
        current_dir = directions[0]

        for d in directions:
            if d != 0 and d != current_dir:
                sign_changes += 1
                current_dir = d

        return sign_changes >= PACING_CROSSINGS

    def _should_fire(self, pet_id: str, event_type: str, now: float) -> bool:
        key = (pet_id, event_type)
        last = self._last_event.get(key)
        if last is not None and now - last < PACING_WINDOW_SECONDS:
            return False
        self._last_event[key] = now
        return True


# ─── Webhook dispatch ────────────────────────────────────────────────────

def dispatch_webhook(pet_id: str, event_type: str, confidence: float) -> bool:
    if not _HTTPX_AVAILABLE:
        print("  [webhook] httpx not installed — skipping dispatch.")
        return False

    event_id = str(uuid.uuid4())
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
        response = httpx.post(
            LARAVEL_WEBHOOK_URL, json=payload, headers=headers, timeout=5.0
        )
        ok = response.status_code in (200, 201)
        status = "OK" if ok else f"HTTP {response.status_code}"
        print(f"  [webhook] {event_type} → {status} (event_id={event_id[:8]}…)")
        return ok
    except httpx.HTTPError as exc:
        print(f"  [webhook] dispatch FAILED: {exc}")
        return False


# ─── Drawing helpers ─────────────────────────────────────────────────────

def scale_zone(frame_w: int, frame_h: int, normalised_zone: np.ndarray) -> np.ndarray:
    pts = normalised_zone.copy()
    pts[:, 0] *= frame_w
    pts[:, 1] *= frame_h
    return pts.astype(np.int32)

def draw_overlay(
    frame: np.ndarray,
    zone: np.ndarray,
    detections: list[dict],
    any_in_zone: bool,
    event_banner: str | None,
    current_scale: float
) -> None:
    zone_colour = (60, 60, 240) if any_in_zone else (220, 200, 40)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], zone_colour)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [zone], isClosed=True, color=zone_colour, thickness=2)
    cv2.putText(
        frame, "DOOR ZONE", (zone[0][0], zone[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_colour, 2,
    )

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cx, cy = (int(v) for v in det["centre"])
        box_colour = (60, 60, 240) if det["in_zone"] else (80, 220, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)
        cv2.circle(frame, (cx, cy), 4, box_colour, -1)
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(
            frame, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 2,
        )

    if current_scale < 1.0:
        cv2.putText(
            frame, f"THERMAL WATCHDOG ACTIVE: Res {int(current_scale*100)}%", 
            (12, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 150, 255), 2,
        )

    if event_banner:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 200), -1)
        cv2.putText(
            frame, event_banner, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )


# ─── Main loop ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="PetPulse Edge AI video runner")
    parser.add_argument("--video", required=True, help="Path to the input .mp4")
    parser.add_argument("--pet-id", required=True, help="Pet UUID for the webhook")
    parser.add_argument("--headless", action="store_true", help="No preview window")
    parser.add_argument("--save-output", default=None, help="Path to save annotated mp4")
    parser.add_argument("--no-webhook", action="store_true", help="Detection only")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    print(f"Loading YOLO26n ONNX Runtime from {MODEL_PATH} …")
    t0 = time.time()
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to load model: {exc}\nEnsure you exported the model to ONNX format.", file=sys.stderr)
        return 1
    print(f"ONNX Model loaded in {time.time() - t0:.1f}s.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}", file=sys.stderr)
        return 1

    orig_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dynamic zone selection based on input filename
    if "catrush" in args.video.lower():
        norm_zone = np.array([[0.05, 0.15], [0.30, 0.15], [0.30, 0.95], [0.05, 0.95]], dtype=np.float32)
    elif "catrunning" in args.video.lower():
        norm_zone = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32)
    else:
        norm_zone = np.array([[0.66, 0.15], [0.97, 0.15], [0.97, 0.95], [0.66, 0.95]], dtype=np.float32)

    zone = scale_zone(orig_frame_w, orig_frame_h, norm_zone)
    tracker = ZoneStateTracker()
    watchdog = AdaptiveThermalWatchdog()

    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_output, fourcc, fps, (orig_frame_w, orig_frame_h))

    frame_idx = 0
    processed = 0
    event_banner: str | None = None
    banner_until = 0.0
    inference_times: list[float] = []

    print("\nProcessing ONNX Edge Inference… (press Q in the preview window to stop)\n")

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if (frame_idx - 1) % args.stride != 0:
            if writer is not None:
                writer.write(original_frame)
            continue

        # Adaptive Resolution Scaling
        current_scale = 1.0
        if inference_times:
            current_scale = watchdog.update_and_get_scale(inference_times[-1] * 1000)
            
        if current_scale < 1.0:
            infer_frame = cv2.resize(original_frame, (0,0), fx=current_scale, fy=current_scale)
        else:
            infer_frame = original_frame

        # Inference
        inf_start = time.time()
        results = model(infer_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        inference_times.append(time.time() - inf_start)
        processed += 1

        detections: list[dict] = []
        best_centroid = None

        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls)]
                if label not in TARGET_CLASSES:
                    continue
                    
                x1, y1, x2, y2 = (float(v) / current_scale for v in box.xyxy[0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                in_zone = cv2.pointPolygonTest(zone, (cx, cy), False) >= 0
                
                detections.append({
                    "label": label,
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2, y2],
                    "centre": [cx, cy],
                    "in_zone": in_zone,
                })
                
                if best_centroid is None:
                    best_centroid = (cx, cy)

        any_in_zone = any(d["in_zone"] for d in detections)
        video_now = frame_idx / fps
        
        # Pass the extracted X-centroid into the upgraded ZoneStateTracker
        pet_cx = best_centroid[0] if best_centroid else None
        event = tracker.update(args.pet_id, any_in_zone, video_now, pet_cx)

        if event is not None:
            event_type, confidence = event
            print(f"[t={video_now:6.1f}s] EVENT: {event_type} (conf {confidence})")
            if not args.no_webhook:
                dispatch_webhook(args.pet_id, event_type, confidence)
            event_banner = f"ALERT: {event_type.upper()} detected"
            banner_until = time.time() + 3.0

        if event_banner and time.time() > banner_until:
            event_banner = None

        draw_overlay(original_frame, zone, detections, any_in_zone, event_banner, current_scale)

        if writer is not None:
            writer.write(original_frame)

        if not args.headless:
            cv2.imshow("PetPulse Edge AI — YOLO26n (ONNX)", original_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        if processed % 50 == 0:
            avg_ms = (sum(inference_times[-50:]) / 50) * 1000
            est_fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"  …{frame_idx}/{total_frames} frames (avg inference {avg_ms:.0f}ms ≈ {est_fps:.1f} FPS) [Scale: {current_scale}]")

    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    if inference_times:
        avg_ms = (sum(inference_times) / len(inference_times)) * 1000
        est_fps = 1000 / avg_ms if avg_ms > 0 else 0
        print(
            f"\n── Summary ──\n"
            f"Processed {processed} frames.\n"
            f"Average inference: {avg_ms:.0f}ms/frame ≈ {est_fps:.1f} FPS.\n"
        )
        if est_fps >= 15:
            print("✓ Meets NFR-PERF-02 (≥15 FPS CPU ONNX inference).")
        else:
            print("⚠ Below the ≥15 FPS NFR-PERF-02 target.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
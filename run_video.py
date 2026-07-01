"""PetPulse Edge AI — Video File Runner (AT3 demonstration entrypoint).

Runs the YOLO26n perception + Zone-Based Logic pipeline against a
pre-recorded .mp4 file on CPU, simulating the edge node. Draws the
door-zone polygon, detection boxes, and live state so the behaviour is
visible on screen (and screen-recordable for the AT3 video), and fires
the real behavioural-events webhook to the Laravel API when an anxiety
event triggers.

This is a standalone entrypoint, separate from the FastAPI service in
app.py, but it reuses identical detection and zone logic so the two
never diverge.

Usage:
    python run_video.py --video samples/dog.mp4 --pet-id <uuid>

    # headless (no preview window, e.g. on a server):
    python run_video.py --video samples/dog.mp4 --pet-id <uuid> --headless

    # save annotated output for the demo video:
    python run_video.py --video samples/dog.mp4 --pet-id <uuid> --save-output out.mp4

Environment (.env or shell):
    YOLO_MODEL_PATH          default: models/yolo26n.pt
    LARAVEL_WEBHOOK_URL      default: http://127.0.0.1:8000/api/v1/behavioral-events
    PETPULSE_WEBHOOK_SECRET  must match the Laravel API's value
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

# httpx is used for the webhook; import lazily so the script can still run
# in detection-only mode if httpx is missing.
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


# ─── Configuration (mirrors app.py) ──────────────────────────────────────

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo26n.pt")
CONFIDENCE_THRESHOLD = 0.25
TARGET_CLASSES = {"dog", "cat"}

LARAVEL_WEBHOOK_URL = os.getenv(
    "LARAVEL_WEBHOOK_URL", "http://127.0.0.1:8000/api/v1/behavioral-events"
)
WEBHOOK_SECRET = os.getenv("PETPULSE_WEBHOOK_SECRET", "change-me-in-env")

# Door-zone polygon (pixel coords). Tuned for a 1280-wide frame; the
# runner scales it to the actual frame size at load time so it lands in a
# sensible place regardless of the source video's resolution.
DOOR_ZONE_NORMALISED = np.array(
    [[0.10, 0.25], [0.23, 0.25], [0.23, 0.80], [0.10, 0.77]], dtype=np.float32
)

PACING_CROSSINGS = 3
PACING_WINDOW_SECONDS = 60.0
PROLONGED_WAIT_SECONDS = 300.0


# ─── Zone state tracker (identical logic to app.py) ──────────────────────

class ZoneStateTracker:
    """Per-pet state for pacing and prolonged-waiting heuristics."""

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
            self._entered_at[pet_id] = now if is_inside else None

        self._was_inside[pet_id] = is_inside

        recent = [
            t for t in self._crossings[pet_id] if now - t <= PACING_WINDOW_SECONDS
        ]
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


# ─── Webhook dispatch (synchronous; this is an offline runner) ───────────

def dispatch_webhook(pet_id: str, event_type: str, confidence: float) -> bool:
    """POST a behavioural event to the Laravel API. Returns success bool."""
    if not _HTTPX_AVAILABLE:
        print("  [webhook] httpx not installed — skipping dispatch (detection-only).")
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
        if not ok:
            print(f"  [webhook] response body: {response.text[:200]}")
        return ok
    except httpx.HTTPError as exc:
        print(f"  [webhook] dispatch FAILED: {exc}")
        return False


# ─── Drawing helpers ─────────────────────────────────────────────────────

def scale_zone(frame_w: int, frame_h: int) -> np.ndarray:
    """Scale the normalised door-zone polygon to actual frame pixels."""
    pts = DOOR_ZONE_NORMALISED.copy()
    pts[:, 0] *= frame_w
    pts[:, 1] *= frame_h
    return pts.astype(np.int32)


def draw_overlay(
    frame: np.ndarray,
    zone: np.ndarray,
    detections: list[dict],
    any_in_zone: bool,
    event_banner: str | None,
) -> None:
    """Draw the door zone, detections, and status onto the frame in place."""
    # Door zone — red when occupied, cyan when clear.
    zone_colour = (60, 60, 240) if any_in_zone else (220, 200, 40)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], zone_colour)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [zone], isClosed=True, color=zone_colour, thickness=2)
    cv2.putText(
        frame, "DOOR ZONE", (zone[0][0], zone[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_colour, 2,
    )

    # Detection boxes.
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

    # Event banner.
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
    parser.add_argument(
        "--no-webhook", action="store_true", help="Detection only; do not POST"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Process every Nth frame (raise to speed up CPU inference)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    # ── Load model ──
    print(f"Loading YOLO26n model from {MODEL_PATH} …")
    t0 = time.time()
    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to load model: {exc}", file=sys.stderr)
        print(
            "If the weights are missing, Ultralytics will auto-download on first\n"
            "use with internet access, or place yolo26n.pt in the models/ folder.",
            file=sys.stderr,
        )
        return 1
    print(f"Model loaded in {time.time() - t0:.1f}s.")

    # ── Open video ──
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}", file=sys.stderr)
        return 1

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_w}x{frame_h} @ {fps:.0f}fps, {total_frames} frames.")

    zone = scale_zone(frame_w, frame_h)
    tracker = ZoneStateTracker()

    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_output, fourcc, fps, (frame_w, frame_h))

    pet_id = args.pet_id
    frame_idx = 0
    processed = 0
    event_banner: str | None = None
    banner_until = 0.0
    inference_times: list[float] = []

    print("\nProcessing… (press Q in the preview window to stop)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Frame striding to speed up CPU inference if needed.
        if (frame_idx - 1) % args.stride != 0:
            if writer is not None:
                writer.write(frame)
            continue

        # ── Inference ──
        inf_start = time.time()
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        inference_times.append(time.time() - inf_start)
        processed += 1

        detections: list[dict] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls)]
                if label not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                in_zone = cv2.pointPolygonTest(zone, (cx, cy), False) >= 0
                detections.append(
                    {
                        "label": label,
                        "confidence": float(box.conf),
                        "bbox": [x1, y1, x2, y2],
                        "centre": [cx, cy],
                        "in_zone": in_zone,
                    }
                )

        any_in_zone = any(d["in_zone"] for d in detections)

        # ── Zone-Based Logic ──
        # Use the video's own timeline (frame_idx / fps) as "now", so the
        # heuristic windows are evaluated against video-time, not wall-time.
        video_now = frame_idx / fps
        event = tracker.update(pet_id, any_in_zone, video_now)

        if event is not None:
            event_type, confidence = event
            print(f"[t={video_now:6.1f}s] EVENT: {event_type} (conf {confidence})")
            if not args.no_webhook:
                dispatch_webhook(pet_id, event_type, confidence)
            event_banner = f"ALERT: {event_type.upper()} detected"
            banner_until = time.time() + 3.0

        # Clear expired banner.
        if event_banner and time.time() > banner_until:
            event_banner = None

        # ── Draw + output ──
        draw_overlay(frame, zone, detections, any_in_zone, event_banner)

        if writer is not None:
            writer.write(frame)

        if not args.headless:
            cv2.imshow("PetPulse Edge AI — YOLO26n", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        # Progress ping every 50 processed frames.
        if processed % 50 == 0:
            avg_ms = (sum(inference_times) / len(inference_times)) * 1000
            est_fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(
                f"  …{frame_idx}/{total_frames} frames "
                f"(avg inference {avg_ms:.0f}ms ≈ {est_fps:.1f} FPS)"
            )

    # ── Cleanup + summary ──
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nAnnotated video saved to {args.save_output}")
    if not args.headless:
        cv2.destroyAllWindows()

    if inference_times:
        avg_ms = (sum(inference_times) / len(inference_times)) * 1000
        est_fps = 1000 / avg_ms if avg_ms > 0 else 0
        print(
            f"\n── Summary ──\n"
            f"Processed {processed} frames.\n"
            f"Average inference: {avg_ms:.0f}ms/frame ≈ {est_fps:.1f} FPS on this CPU.\n"
        )
        # NFR-PERF-02 reality check.
        if est_fps >= 15:
            print("✓ Meets NFR-PERF-02 (≥15 FPS CPU inference).")
        else:
            print(
                f"⚠ Below the ≥15 FPS NFR-PERF-02 target. Options: raise --stride, "
                f"export the model to ONNX, or document the CPU-bound result honestly."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import mimetypes
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "inference_logs.sqlite3"
MODEL_DIR = PROJECT_ROOT / "saved_models"
MODEL_PATH = MODEL_DIR / "vehicle_classifier_final.keras"
CLASS_JSON_PATH = MODEL_DIR / "class_names.json"
SAMPLE_DIR = PROJECT_ROOT / "test"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_PORT = 8000

_model = None
_model_lock = threading.Lock()
_model_load_error = None
_runtime_modules = None
_runtime_error = None


def safe_print(message: str = "") -> None:
    try:
        if sys.stdout:
            print(message, flush=True)
    except Exception:
        pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_label(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ").title()


def read_class_config() -> dict:
    if not CLASS_JSON_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {CLASS_JSON_PATH}")
    with CLASS_JSON_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("img_size", 128)
    payload.setdefault("class_names", [])
    return payload


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT,
                confidence REAL,
                latency_ms INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                status TEXT NOT NULL,
                message TEXT,
                probabilities_json TEXT
            )
            """
        )
        conn.commit()


def log_inference(
    *,
    filename: str,
    status: str,
    predicted_class: str | None = None,
    confidence: float | None = None,
    latency_ms: int | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    message: str | None = None,
    probabilities: list[dict] | None = None,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO inference_logs (
                timestamp, filename, predicted_class, confidence, latency_ms,
                image_width, image_height, status, message, probabilities_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                filename,
                predicted_class,
                confidence,
                latency_ms,
                image_width,
                image_height,
                status,
                message,
                json.dumps(probabilities or []),
            ),
        )
        conn.commit()


def runtime_status() -> dict:
    tensorflow_available = importlib.util.find_spec("tensorflow") is not None
    numpy_available = importlib.util.find_spec("numpy") is not None
    pillow_available = importlib.util.find_spec("PIL") is not None
    return {
        "tensorflow": tensorflow_available,
        "numpy": numpy_available,
        "pillow": pillow_available,
        "ready_for_inference": tensorflow_available and numpy_available,
    }


def get_runtime_modules():
    global _runtime_modules, _runtime_error
    if _runtime_modules is not None:
        return _runtime_modules
    if _runtime_error is not None:
        raise RuntimeError(_runtime_error)
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on local runtime
        _runtime_error = str(exc)
        raise RuntimeError(_runtime_error) from exc
    _runtime_modules = {"np": np, "tf": tf}
    return _runtime_modules


def get_model():
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        raise RuntimeError(_model_load_error)
    with _model_lock:
        if _model is not None:
            return _model
        try:
            modules = get_runtime_modules()
            tf = modules["tf"]
            _model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            _model_load_error = str(exc)
            raise RuntimeError(_model_load_error) from exc
    return _model


def detected_image_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return None


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "uploaded_image").name.strip()
    return name or "uploaded_image"


def validate_upload(filename: str, image_bytes: bytes) -> tuple[bool, str]:
    if not image_bytes:
        return False, "The uploaded file was empty."
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return False, "The image is larger than the 8 MB local upload limit."
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, "Use a JPG, JPEG, or PNG image."
    signature = detected_image_type(image_bytes)
    if signature is None:
        return False, "The file signature does not look like a JPG or PNG image."
    if signature == ".png" and extension != ".png":
        return False, "The image content is PNG, but the filename extension is not .png."
    if signature == ".jpg" and extension not in {".jpg", ".jpeg"}:
        return False, "The image content is JPEG, but the filename extension is not .jpg or .jpeg."
    return True, "ok"


def predict_image(image_bytes: bytes, filename: str) -> dict:
    started = time.perf_counter()
    class_config = read_class_config()
    class_names = class_config["class_names"]
    img_size = int(class_config.get("img_size", 128))
    modules = get_runtime_modules()
    np = modules["np"]
    tf = modules["tf"]
    model = get_model()

    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    original_shape = tf.shape(image).numpy().tolist()
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    batch = tf.expand_dims(image, axis=0)

    probs = model(batch, training=False).numpy()[0]
    top_indices = np.argsort(probs)[::-1]
    latency_ms = int((time.perf_counter() - started) * 1000)

    probabilities = []
    for idx, score in enumerate(probs):
        class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        probabilities.append(
            {
                "class_name": class_name,
                "display_name": display_label(class_name),
                "probability": float(score),
            }
        )

    top_k = []
    for idx in top_indices[:5]:
        class_name = class_names[int(idx)] if int(idx) < len(class_names) else f"class_{idx}"
        top_k.append(
            {
                "class_name": class_name,
                "display_name": display_label(class_name),
                "probability": float(probs[int(idx)]),
            }
        )

    top_class = top_k[0]["class_name"]
    top_confidence = top_k[0]["probability"]
    image_height, image_width = int(original_shape[0]), int(original_shape[1])

    log_inference(
        filename=filename,
        status="success",
        predicted_class=top_class,
        confidence=top_confidence,
        latency_ms=latency_ms,
        image_width=image_width,
        image_height=image_height,
        message="Prediction completed.",
        probabilities=probabilities,
    )

    return {
        "ok": True,
        "prediction": {
            "label": top_class,
            "display_label": display_label(top_class),
            "confidence": top_confidence,
            "top_k": top_k,
        },
        "probabilities": probabilities,
        "image": {
            "filename": filename,
            "width": image_width,
            "height": image_height,
            "size_bytes": len(image_bytes),
        },
        "latency_ms": latency_ms,
        "model": {
            "path": str(MODEL_PATH),
            "img_size": img_size,
            "num_classes": len(class_names),
        },
    }


def query_history(limit: int = 50, search: str = "") -> list[dict]:
    limit = max(1, min(limit, 250))
    params: list[object] = []
    where = ""
    if search:
        like = f"%{search}%"
        where = """
            WHERE filename LIKE ?
               OR predicted_class LIKE ?
               OR status LIKE ?
               OR message LIKE ?
        """
        params.extend([like, like, like, like])
    params.append(limit)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, timestamp, filename, predicted_class, confidence,
                   latency_ms, image_width, image_height, status, message
            FROM inference_logs
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def list_sample_images(limit: int = 12) -> list[dict]:
    if not SAMPLE_DIR.exists():
        return []
    samples = []
    with os.scandir(SAMPLE_DIR) as entries:
        for entry in entries:
            if len(samples) >= limit:
                break
            if not entry.is_file():
                continue
            ext = Path(entry.name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            size = entry.stat().st_size
            samples.append(
                {
                    "filename": entry.name,
                    "url": f"/samples/{quote(entry.name)}",
                    "size_bytes": size,
                }
            )
    return samples


def read_static_file(path: Path) -> tuple[bytes, str]:
    content = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return content, mime_type


class IntelliTrafficHandler(BaseHTTPRequestHandler):
    server_version = "IntelliTraffic/1.0"

    def log_message(self, fmt: str, *args) -> None:
        try:
            if sys.stdout:
                sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        except Exception:
            pass

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, mime_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str, **extra) -> None:
        payload = {"ok": False, "error": message}
        payload.update(extra)
        self.send_json(payload, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            path = "/index.html"

        if path == "/api/status":
            self.handle_status()
            return
        if path == "/api/history":
            self.handle_history(parsed.query)
            return
        if path == "/api/history.csv":
            self.handle_history_csv(parsed.query)
            return
        if path == "/api/samples":
            self.send_json({"ok": True, "samples": list_sample_images()})
            return
        if path.startswith("/samples/"):
            self.handle_sample(path)
            return
        if path.startswith("/"):
            self.handle_static(path)
            return

        self.send_error_json(HTTPStatus.NOT_FOUND, "Route not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            self.handle_predict()
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "Route not found.")

    def handle_static(self, path: str) -> None:
        requested = unquote(path.lstrip("/"))
        static_path = (STATIC_DIR / requested).resolve()
        try:
            static_path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error_json(HTTPStatus.FORBIDDEN, "Invalid static path.")
            return
        if not static_path.exists() or not static_path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "Static file not found.")
            return
        body, mime_type = read_static_file(static_path)
        self.send_bytes(body, mime_type)

    def handle_sample(self, path: str) -> None:
        filename = unquote(path.split("/samples/", 1)[1])
        if Path(filename).name != filename:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid sample filename.")
            return
        sample_path = (SAMPLE_DIR / filename).resolve()
        try:
            sample_path.relative_to(SAMPLE_DIR.resolve())
        except ValueError:
            self.send_error_json(HTTPStatus.FORBIDDEN, "Invalid sample path.")
            return
        if not sample_path.exists() or not sample_path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "Sample not found.")
            return
        body, mime_type = read_static_file(sample_path)
        self.send_bytes(body, mime_type)

    def handle_status(self) -> None:
        try:
            class_config = read_class_config()
            class_names = class_config.get("class_names", [])
            class_error = None
        except Exception as exc:
            class_config = {"img_size": 128}
            class_names = []
            class_error = str(exc)
        payload = {
            "ok": True,
            "app": "IntelliTraffic",
            "project_root": str(PROJECT_ROOT),
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "class_json_path": str(CLASS_JSON_PATH),
            "class_json_exists": CLASS_JSON_PATH.exists(),
            "class_error": class_error,
            "img_size": int(class_config.get("img_size", 128)),
            "num_classes": len(class_names),
            "class_names": class_names,
            "database_path": str(DB_PATH),
            "sample_dir_exists": SAMPLE_DIR.exists(),
            "runtime": runtime_status(),
            "model_loaded": _model is not None,
            "model_load_error": _model_load_error,
            "runtime_error": _runtime_error,
        }
        self.send_json(payload)

    def handle_history(self, query: str) -> None:
        params = parse_qs(query)
        search = params.get("q", [""])[0].strip()
        try:
            limit = int(params.get("limit", ["50"])[0])
        except ValueError:
            limit = 50
        self.send_json({"ok": True, "logs": query_history(limit=limit, search=search)})

    def handle_history_csv(self, query: str) -> None:
        params = parse_qs(query)
        search = params.get("q", [""])[0].strip()
        rows = query_history(limit=250, search=search)
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "id",
                "timestamp",
                "filename",
                "predicted_class",
                "confidence",
                "latency_ms",
                "image_width",
                "image_height",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        body = output.getvalue().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", "attachment; filename=intellitraffic_history.csv")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_predict(self) -> None:
        try:
            filename, image_bytes = self.parse_upload()
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        filename = sanitize_filename(filename)
        is_valid, validation_message = validate_upload(filename, image_bytes)
        if not is_valid:
            log_inference(
                filename=filename,
                status="rejected",
                message=validation_message,
            )
            self.send_error_json(HTTPStatus.BAD_REQUEST, validation_message)
            return

        try:
            payload = predict_image(image_bytes, filename)
        except Exception as exc:
            log_inference(
                filename=filename,
                status="error",
                message=str(exc),
            )
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Inference failed. Check the model runtime and try again.",
                detail=str(exc),
            )
            return

        preview = base64.b64encode(image_bytes[: min(len(image_bytes), 256)]).decode("ascii")
        payload["image"]["preview_hash"] = preview[:24]
        self.send_json(payload)

    def parse_upload(self) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        content_length = self.headers.get("Content-Length")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Expected a multipart image upload.")
        if not content_length:
            raise ValueError("Missing Content-Length header.")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length <= 0:
            raise ValueError("The upload body is empty.")
        if length > MAX_UPLOAD_BYTES + 4096:
            raise ValueError("The upload is larger than the local limit.")

        body = self.rfile.read(length)
        mime_message = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + body
        message = BytesParser(policy=policy.default).parsebytes(mime_message)
        if not message.is_multipart():
            raise ValueError("Could not parse multipart upload.")
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            if "form-data" not in disposition:
                continue
            field_name = part.get_param("name", header="content-disposition")
            if field_name != "image":
                continue
            filename = part.get_filename() or "uploaded_image"
            payload = part.get_payload(decode=True) or b""
            return filename, payload
        raise ValueError("Upload field 'image' was not found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IntelliTraffic local web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    server = ThreadingHTTPServer((args.host, args.port), IntelliTrafficHandler)
    safe_print("IntelliTraffic local app")
    safe_print(f"URL: http://{args.host}:{args.port}")
    safe_print(f"Model: {MODEL_PATH}")
    safe_print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        safe_print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

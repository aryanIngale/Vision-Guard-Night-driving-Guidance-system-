"""
Enhanced_App.py — Night Guidance System with Multi-Class Detection
Flask backend with enhanced object detection and danger alerts

"""

import os
import sys
import uuid
import threading
import time

from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, send_file,
)
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        FLASK_HOST, FLASK_PORT, FLASK_DEBUG, MAX_CONTENT_MB,
        UPLOAD_DIR, OUTPUT_DIR, BEST_MODEL_PATH,
        CONF_THRESHOLD, NIGHT_ENHANCE, CLAHE_CLIP,
    )
except ImportError:
    # Fallback defaults
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG = "0.0.0.0", 5000, True
    MAX_CONTENT_MB = 500
    UPLOAD_DIR, OUTPUT_DIR = "uploads", "outputs"
    BEST_MODEL_PATH = "best_model.pth"
    CONF_THRESHOLD, NIGHT_ENHANCE, CLAHE_CLIP = 0.5, True, 2.0

# Import inference engines
try:
    from Enhanced_Inference import load_model, process_video
    ENHANCED_AVAILABLE = True
except ImportError:
    print("[App] Enhanced_Inference not found, falling back to standard")
    from Inference import load_model, process_video
    ENHANCED_AVAILABLE = False


# App setup


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

app = Flask(
    __name__,
    template_folder=os.path.abspath(TEMPLATE_DIR) if os.path.exists(os.path.abspath(TEMPLATE_DIR)) else None,
    static_folder=os.path.abspath(STATIC_DIR) if os.path.exists(os.path.abspath(STATIC_DIR)) else None,
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# Job store


jobs: dict = {}
# job = {
#   "status": "pending" | "processing" | "done" | "error",
#   "progress": 0–100,
#   "stats": {...},
#   "input_file": filename,
#   "output_file": filename,
#   "error": str | None,
# }


# Runtime settings


runtime = {
    "threshold": CONF_THRESHOLD,
    "night_enhance": NIGHT_ENHANCE,
    "clahe_clip": CLAHE_CLIP,
    "use_enhanced_model": ENHANCED_AVAILABLE,
    "show_danger_zones": True,
}


# Helpers


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_model():
    """Lazy-load and cache the model"""
    use_enhanced = runtime.get("use_enhanced_model", False) and ENHANCED_AVAILABLE
    model, device = load_model(BEST_MODEL_PATH, use_enhanced=use_enhanced)
    return model, device



# Background processing


def _process_job(job_id: str):
    job = jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 0

    try:
        model, device = get_model()

        in_path = os.path.join(UPLOAD_DIR, job["input_file"])
        out_name = f"processed_{job_id}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        def progress_cb(pct: float):
            job["progress"] = round(pct, 1)

        stats = process_video(
            model, device,
            input_path=in_path,
            output_path=out_path,
            threshold=runtime["threshold"],
            night_enhance=runtime["night_enhance"],
            clahe_clip=runtime["clahe_clip"],
            progress_cb=progress_cb,
            show_danger_zones=runtime["show_danger_zones"],
        )

        job["output_file"] = stats["output_file"]
        job["stats"] = stats
        job["status"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[Job {job_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()



# Routes


@app.route("/")
def index():
    # Serve index.html from uploads if templates don't exist
    if os.path.exists("/mnt/user-data/uploads/index.html"):
        return send_file("/mnt/user-data/uploads/index.html")
    return render_template("index.html")


@app.route("/video/<filename>")
def serve_video(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(f"{job_id}_{f.filename}")
    f.save(os.path.join(UPLOAD_DIR, filename))

    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "stats": {},
        "input_file": filename,
        "output_file": None,
        "error": None,
        "created_at": time.time(),
    }

    return jsonify({"job_id": job_id, "filename": f.filename}), 200


@app.route("/api/process/<job_id>", methods=["POST"])
def api_process(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["status"] in ("processing", "done"):
        return jsonify({"error": "Already processed or processing"}), 400

    # Apply per-request overrides
    body = request.get_json(silent=True) or {}
    if "threshold" in body:
        runtime["threshold"] = float(body["threshold"])
    if "night_enhance" in body:
        runtime["night_enhance"] = bool(body["night_enhance"])
    if "clahe_clip" in body:
        runtime["clahe_clip"] = float(body["clahe_clip"])
    if "use_enhanced_model" in body:
        runtime["use_enhanced_model"] = bool(body["use_enhanced_model"]) and ENHANCED_AVAILABLE
    if "show_danger_zones" in body:
        runtime["show_danger_zones"] = bool(body["show_danger_zones"])

    t = threading.Thread(target=_process_job, args=(job_id,), daemon=True)
    t.start()

    return jsonify({"message": "Processing started", "job_id": job_id}), 200


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
    })


@app.route("/api/result/<job_id>")
def api_result(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Not ready yet"}), 400

    return jsonify({
        "job_id": job_id,
        "video_url": f"/video/{job['output_file']}",
        "stats": job["stats"],
    })


@app.route("/api/model_info")
def api_model_info():
    exists = os.path.exists(BEST_MODEL_PATH)
    size = os.path.getsize(BEST_MODEL_PATH) / 1e6 if exists else 0

    return jsonify({
        "model_path": BEST_MODEL_PATH,
        "model_exists": exists,
        "model_size_mb": round(size, 2),
        "enhanced_available": ENHANCED_AVAILABLE,
        "runtime": runtime,
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    body = request.get_json(silent=True) or {}
    
    if "threshold" in body:
        runtime["threshold"] = float(body["threshold"])
    if "night_enhance" in body:
        runtime["night_enhance"] = bool(body["night_enhance"])
    if "clahe_clip" in body:
        runtime["clahe_clip"] = float(body["clahe_clip"])
    if "use_enhanced_model" in body:
        runtime["use_enhanced_model"] = bool(body["use_enhanced_model"]) and ENHANCED_AVAILABLE
    if "show_danger_zones" in body:
        runtime["show_danger_zones"] = bool(body["show_danger_zones"])
    
    return jsonify({"settings": runtime})


@app.route("/api/use_enhanced", methods=["POST"])
def api_use_enhanced():
    """Toggle enhanced model"""
    body = request.get_json(silent=True) or {}
    use_enhanced = body.get("enabled", True)
    
    if use_enhanced and not ENHANCED_AVAILABLE:
        return jsonify({
            "error": "Enhanced model not available",
            "enhanced_available": False
        }), 400
    
    runtime["use_enhanced_model"] = use_enhanced
    
    return jsonify({
        "enabled": runtime["use_enhanced_model"],
        "message": "Enhanced model " + ("enabled" if use_enhanced else "disabled")
    })


@app.route("/api/detection_stats")
def api_detection_stats():
    """Get aggregated detection statistics across all jobs"""
    total_jobs = len(jobs)
    completed = sum(1 for j in jobs.values() if j["status"] == "done")
    
    total_vehicles = 0
    total_pedestrians = 0
    total_dangers = 0
    
    for job in jobs.values():
        if job["status"] == "done" and "stats" in job:
            stats = job["stats"]
            total_vehicles += stats.get("total_vehicles", 0)
            total_pedestrians += stats.get("total_pedestrians", 0)
            total_dangers += stats.get("danger_count", 0)
    
    return jsonify({
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "total_vehicles_detected": total_vehicles,
        "total_pedestrians_detected": total_pedestrians,
        "total_danger_events": total_dangers,
    })



# Run


if __name__ == "__main__":
    print(f"\n Enhanced Driving Guidance System — Web UI")
    print(f" Open  →  http://localhost:{FLASK_PORT}")
    print(f" Enhanced Model: {'Available' if ENHANCED_AVAILABLE else 'Not Available'}\n")

    # Warm up model
    def _warmup():
        try:
            get_model()
            print("  [App] Model warmed up and ready.")
        except Exception as e:
            print(f"  [App] Model warmup failed: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
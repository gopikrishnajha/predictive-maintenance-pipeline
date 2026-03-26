#!/usr/bin/env python3
"""
PACE Web UI - Flask backend
Provides:
  1. Pipeline runner with real-time SSE output streaming
  2. Interactive chat (analysis, evidence, SQL agents)
"""

import sys
import json
import logging
import subprocess
import threading
import queue
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response

# Project root is one level up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline runner state
# ---------------------------------------------------------------------------
pipeline_lock = threading.Lock()
pipeline_running = False
pipeline_output_queues: list[queue.Queue] = []
pipeline_output_queues_lock = threading.Lock()


def _broadcast(line: str):
    """Push a line to every connected SSE listener."""
    with pipeline_output_queues_lock:
        for q in pipeline_output_queues:
            q.put(line)


def _load_full_config():
    """Load the full YAML config file for the active use case."""
    try:
        with open(PROJECT_ROOT / "config.json") as f:
            main_cfg = json.load(f)
        use_case = main_cfg.get("use-case-id", "pipeline_defects_detection")
        cfg_path = PROJECT_ROOT / "config" / f"{use_case}.yaml"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Config load failed, falling back to empty config", exc_info=True)
    return {}


def _load_inference_config():
    """Load inference config from the YAML config file."""
    return _load_full_config().get("inference", {})


def _run_pipeline(input_mode: str, num_images: int, device: str,
                  inference_interval: int = 1, video_path: str = None):
    """Run the complete pipeline in a background thread and broadcast output."""
    global pipeline_running
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_complete_pipeline.py"),
            "--device", device,
        ]
        if input_mode == "video":
            if video_path:
                cmd.extend(["--video", video_path])
            if inference_interval > 1:
                cmd.extend(["--inference-interval", str(inference_interval)])
        else:
            cmd.extend(["--num-images", str(num_images)])
            # Pass empty video to force image mode
            # (run_complete_pipeline defaults to video; override with no --video)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        for line in proc.stdout:
            _broadcast(line)

        proc.wait()
        if proc.returncode == 0:
            _broadcast("\n[PIPELINE COMPLETE ✅]\n")
        else:
            _broadcast(f"\n[PIPELINE FAILED with exit code {proc.returncode}]\n")
    except Exception as exc:
        _broadcast(f"\n[ERROR] {exc}\n")
    finally:
        _broadcast("[DONE]")
        with pipeline_lock:
            pipeline_running = False


# ---------------------------------------------------------------------------
# Chat engine (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_chat_instance = None
_chat_lock = threading.Lock()
_chat_loading = False
_chat_error = None


def _get_chat():
    """Return the shared InteractiveChat instance, creating it on first call."""
    global _chat_instance, _chat_loading, _chat_error
    if _chat_instance is not None:
        return _chat_instance

    with _chat_lock:
        if _chat_instance is not None:
            return _chat_instance
        _chat_loading = True
        _chat_error = None
        try:
            import sqlite3
            from interactive_chat import InteractiveChat
            chat = InteractiveChat()
            if not chat.load_config():
                raise RuntimeError("Failed to load chat config / LLM")
            chat.load_artifacts()
            # Re-open SQLite connection with check_same_thread=False so it
            # can be used from Flask's request-handling threads.
            if chat.db_client and chat.db_client.conn:
                chat.db_client.conn.close()
                chat.db_client.conn = sqlite3.connect(
                    str(chat.db_client.db_path), check_same_thread=False
                )
                chat.db_client.conn.row_factory = sqlite3.Row
            _chat_instance = chat
            return _chat_instance
        except Exception as e:
            _chat_error = str(e)
            raise
        finally:
            _chat_loading = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    resp = app.make_response(render_template("index.html"))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route("/api/predef-questions")
def predef_questions():
    """Return predefined questions from web_app/predef_questions.json."""
    qfile = Path(__file__).resolve().parent / "predef_questions.json"
    try:
        with open(qfile, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"analysis": [], "evidence": [], "sql": []})


@app.route("/api/agent-output/<agent>")
def agent_output(agent):
    """Return the text content of an agent output file."""
    file_map = {
        "analysis": PROJECT_ROOT / "out" / "agent" / "analysis_summary.txt",
        "evidence": PROJECT_ROOT / "out" / "agent" / "evidence_trail.txt",
    }
    path = file_map.get(agent)
    if not path or not path.exists():
        return jsonify({"text": None})
    try:
        return jsonify({"text": path.read_text(encoding="utf-8")})
    except Exception:
        return jsonify({"text": None})


@app.route("/api/pipeline/start", methods=["POST"])
def pipeline_start():
    global pipeline_running
    with pipeline_lock:
        if pipeline_running:
            return jsonify({"ok": False, "error": "Pipeline is already running"}), 409
        pipeline_running = True

    data = request.get_json(silent=True) or {}
    inf_cfg = _load_inference_config()

    input_mode = data.get("input_mode", inf_cfg.get("input_mode", "video"))
    num_images = int(data.get("num_images", 100))
    device = data.get("device", inf_cfg.get("device", "GPU"))
    inference_interval = int(data.get("inference_interval",
                                       inf_cfg.get("inference_interval", 1)))
    video_path = data.get("video_path",
                          inf_cfg.get("video_path",
                                      "datasets/pipeline_defects_detection/video/input.mp4"))

    t = threading.Thread(
        target=_run_pipeline,
        args=(input_mode, num_images, device, inference_interval, video_path),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True})


@app.route("/api/config")
def get_config():
    """Return inference config defaults for the UI."""
    cfg = _load_inference_config()
    return jsonify({
        "input_mode": cfg.get("input_mode", "video"),
        "device": cfg.get("device", "GPU"),
        "inference_interval": cfg.get("inference_interval", 1),
        "video_path": cfg.get("video_path",
                              "datasets/pipeline_defects_detection/video/input.mp4"),
        "images_path": cfg.get("images_path",
                               "datasets/pipeline_defects_detection/val_seq"),
    })


@app.route("/api/pipeline/stream")
def pipeline_stream():
    """SSE endpoint – streams pipeline stdout to the browser."""
    q: queue.Queue = queue.Queue()
    with pipeline_output_queues_lock:
        pipeline_output_queues.append(q)

    def generate():
        try:
            while True:
                try:
                    line = q.get(timeout=30)
                except queue.Empty:
                    yield ":\n\n"  # keep-alive
                    continue
                if line == "[DONE]":
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
                yield f"data: {json.dumps({'line': line})}\n\n"
        finally:
            with pipeline_output_queues_lock:
                pipeline_output_queues.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/pipeline/status")
def pipeline_status():
    return jsonify({"running": pipeline_running})


@app.route("/api/chat/status")
def chat_status():
    """Return whether the chat engine is loaded / loading."""
    return jsonify({
        "loaded": _chat_instance is not None,
        "loading": _chat_loading,
        "error": _chat_error,
    })


@app.route("/api/chat/init", methods=["POST"])
def chat_init():
    """Eagerly initialise the chat engine."""
    try:
        _get_chat()
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Chat init failed")
        return jsonify({"ok": False, "error": "Chat initialization failed"}), 500


@app.route("/api/chat/ask", methods=["POST"])
def chat_ask():
    """
    Expects JSON: {mode: "analysis"|"evidence"|"sql", question: "..."}
    Returns JSON: {answer: "..."}
    """
    data = request.get_json(force=True)
    mode = data.get("mode", "analysis")
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        chat = _get_chat()
    except Exception as e:
        logger.exception("Chat engine not ready")
        return jsonify({"error": "Chat engine not ready"}), 503

    try:
        if mode == "analysis":
            if not chat.analysis_summary:
                return jsonify({"error": "No analysis artifacts found. Run the pipeline first."}), 400
            prompt = f"Context:\n{chat.analysis_summary}\n\nQuestion: {question}\n\n{chat.QA_INSTRUCTION}"
            answer = chat.ask_question(prompt, show_thinking=False)

        elif mode == "evidence":
            evidence_trail = ""
            trail_path = PROJECT_ROOT / "out" / "agent" / "evidence_trail.txt"
            if trail_path.exists():
                with open(trail_path, "r") as f:
                    evidence_trail = f.read()
            prompt = f"Context:\n{evidence_trail}\n\nQuestion: {question}\n\n{chat.QA_INSTRUCTION}"
            answer = chat.ask_question(prompt, show_thinking=False)

        elif mode == "sql":
            sql_query = chat.generate_sql_query(question)
            results = chat.execute_and_format_query(sql_query)
            answer = f"Generated SQL:\n{sql_query}\n\n{results}"

            # Extract frame_ids from raw query results for frame display
            frame_ids = []
            try:
                raw = chat.db_client.execute_query(sql_query)
                if raw:
                    cols = list(raw[0].keys())
                    if 'frame_id' in cols:
                        seen = set()
                        for row in raw:
                            fid = row['frame_id']
                            if fid not in seen:
                                seen.add(fid)
                                frame_ids.append(fid)
            except Exception:
                logger.debug("SQL query for frame IDs failed; non-critical", exc_info=True)

            return jsonify({"answer": answer, "frame_ids": frame_ids[:50]})

        else:
            return jsonify({"error": f"Unknown mode: {mode}"}), 400

        return jsonify({"answer": answer})

    except Exception as e:
        logger.exception("Chat request failed")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/visualizations")
def list_visualizations():
    """Return a list of visualization image paths."""
    viz_dir = PROJECT_ROOT / "out" / "viz"
    if not viz_dir.exists():
        return jsonify({"images": []})
    images = sorted(
        [p.name for p in viz_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
    )
    return jsonify({"images": images})


@app.route("/viz/<path:filename>")
def serve_viz(filename):
    """Serve visualization images."""
    from flask import send_from_directory
    return send_from_directory(str(PROJECT_ROOT / "out" / "viz"), filename)


@app.route("/api/annotated-video")
def annotated_video_info():
    """Check if annotated video exists."""
    vid = PROJECT_ROOT / "out" / "annotated_output.mp4"
    if vid.exists():
        return jsonify({"exists": True,
                        "path": "/video/annotated_output.mp4",
                        "size_mb": round(vid.stat().st_size / 1e6, 1)})
    return jsonify({"exists": False, "path": None, "size_mb": 0})


@app.route("/video/<path:filename>")
def serve_video(filename):
    """Serve output videos."""
    from flask import send_from_directory
    return send_from_directory(str(PROJECT_ROOT / "out"), filename)


# ---------------------------------------------------------------------------
# Ticketing
# ---------------------------------------------------------------------------
@app.route("/api/ticket/create", methods=["POST"])
def ticket_create():
    """Create a ticket for a specific frame."""
    data = request.get_json(force=True)
    frame_id = data.get("frame_id")
    if frame_id is None:
        return jsonify({"ok": False, "error": "Missing frame_id"}), 400

    try:
        from src.agents.ticketing_agent import create_ticket
        from src.utility.sqlite_client import SQLiteClient

        full_cfg = _load_full_config()
        sqlite_cfg = full_cfg.get('sqlite', {})
        db_path = sqlite_cfg.get('db_path', 'out/sql_data/detections.db')
        db = SQLiteClient(db_path=db_path)
        result = create_ticket(db, int(frame_id))
        db.close()
        return jsonify(result)
    except Exception as e:
        logger.exception("Ticket creation failed for frame %s", frame_id)
        return jsonify({"ok": False, "error": "Ticket creation failed"}), 500


@app.route("/api/tickets")
def list_tickets():
    """List all available tickets."""
    tickets_dir = PROJECT_ROOT / "out" / "tickets"
    tickets = []
    if tickets_dir.exists():
        for f in sorted(tickets_dir.glob("ticket_*.html")):
            # Extract frame_id from filename ticket_<frame_id>.html
            try:
                fid = int(f.stem.replace("ticket_", ""))
            except ValueError:
                fid = None
            tickets.append({
                "filename": f.name,
                "frame_id": fid,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return jsonify({"tickets": tickets})


@app.route("/ticket/<filename>")
def serve_ticket(filename):
    """Serve a ticket HTML file."""
    from flask import send_from_directory
    tickets_dir = PROJECT_ROOT / "out" / "tickets"
    return send_from_directory(str(tickets_dir), filename)


# ---------------------------------------------------------------------------
def run_server(host="127.0.0.1", port=5000, debug=False):
    """Entry point used by the launch script."""
    import socket
    # Allow immediate port reuse after stop/restart (avoid Address already in use)
    socket.setdefaulttimeout(5)
    print(f"\n{'='*60}")
    print(f"  PACE Web UI  →  http://localhost:{port}")
    print(f"{'='*60}\n")
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PACE Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug)

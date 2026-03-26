"""
Ticketing Agent - Generate HTML tickets for defect frames.

Creates self-contained HTML tickets with detection details,
evidence trail context, and embedded frame images.
"""

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_evidence_trail() -> str:
    """Read the evidence trail file if it exists."""
    trail_path = PROJECT_ROOT / "out" / "agent" / "evidence_trail.txt"
    if trail_path.exists():
        return trail_path.read_text(encoding="utf-8")
    return "(No evidence trail available)"


def _encode_image_base64(image_path: Path) -> Optional[str]:
    """Encode an image to base64 for embedding in HTML."""
    if image_path.exists():
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    return None


def _get_frame_detections(db_client, frame_id: int) -> List[Dict[str, Any]]:
    """Get all detections for a specific frame from the database."""
    try:
        rows = db_client.execute_query(
            "SELECT * FROM detections WHERE frame_id = ?", (frame_id,)
        )
        return rows
    except Exception:
        return []


def generate_ticket_html(
    frame_id: int,
    detections: List[Dict[str, Any]],
    evidence_trail: str,
    image_path: Path,
    image_base64: Optional[str],
) -> str:
    """Generate a self-contained HTML ticket for a defect frame."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build detections table rows
    det_rows = ""
    for det in detections:
        det_rows += f"""
        <tr>
            <td>{det.get('label', 'N/A')}</td>
            <td>{det.get('confidence', 0):.3f}</td>
            <td>({det.get('x', 0)}, {det.get('y', 0)})</td>
            <td>{det.get('width', 0)} × {det.get('height', 0)}</td>
        </tr>"""

    # Image tag (embedded base64 or path reference)
    if image_base64:
        img_tag = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Frame {frame_id}" style="max-width:100%;border-radius:8px;border:1px solid #333">'
    else:
        img_tag = f'<p style="color:#f44">Image not found: {image_path}</p>'

    # Escape evidence trail for HTML
    evidence_html = (
        evidence_trail.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Defect Ticket — Frame {frame_id}</title>
<style>
  :root {{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#c9d1d9;--accent:#58a6ff;--warn:#f85149}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:32px;line-height:1.6}}
  .ticket{{max-width:900px;margin:0 auto;background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
  .header{{background:linear-gradient(135deg,#1a2332,#0d1117);padding:24px 32px;border-bottom:1px solid var(--border)}}
  .header h1{{color:var(--accent);font-size:1.5rem;margin-bottom:4px}}
  .header .meta{{color:#8b949e;font-size:.85rem}}
  .section{{padding:20px 32px;border-bottom:1px solid var(--border)}}
  .section:last-child{{border-bottom:none}}
  .section h2{{color:var(--accent);font-size:1.1rem;margin-bottom:12px;display:flex;align-items:center;gap:8px}}
  table{{width:100%;border-collapse:collapse;margin-top:8px}}
  th,td{{text-align:left;padding:8px 12px;border:1px solid var(--border);font-size:.88rem}}
  th{{background:#0d1117;color:var(--accent);font-weight:600}}
  td{{color:var(--text)}}
  .evidence{{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:16px;
    font-family:'JetBrains Mono',monospace;font-size:.8rem;white-space:pre-wrap;max-height:400px;overflow-y:auto;color:#8b949e}}
  .img-container{{text-align:center;margin-top:8px}}
  .path-info{{color:#8b949e;font-size:.8rem;margin-top:8px;font-family:monospace}}
  @media print{{body{{background:#fff;color:#000}} .ticket{{border:2px solid #000}} th{{background:#eee;color:#000}} td{{color:#000}}}}
</style>
</head>
<body>
<div class="ticket">
  <div class="header">
    <h1>🎫 Defect Ticket — Frame {frame_id}</h1>
    <div class="meta">Generated: {timestamp} &nbsp;|&nbsp; Ticket ID: TICKET-F{frame_id:06d}</div>
  </div>

  <div class="section">
    <h2>📸 Annotated Frame</h2>
    <div class="img-container">{img_tag}</div>
    <div class="path-info">Image path: {image_path}</div>
  </div>

  <div class="section">
    <h2>🔍 Detection Details ({len(detections)} detection{'s' if len(detections) != 1 else ''})</h2>
    <table>
      <thead><tr><th>Label</th><th>Confidence</th><th>Position (x, y)</th><th>Size (w × h)</th></tr></thead>
      <tbody>{det_rows if det_rows else '<tr><td colspan="4" style="text-align:center;color:#8b949e">No detections found</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>📋 Evidence Trail</h2>
    <div class="evidence">{evidence_html}</div>
  </div>
</div>
</body>
</html>"""
    return html


def create_ticket(db_client, frame_id: int) -> Dict[str, Any]:
    """
    Create a ticket for a specific frame.

    Args:
        db_client: SQLiteClient instance for querying detections
        frame_id: The frame ID to create a ticket for

    Returns:
        Dict with 'ok', 'path', and 'message' keys
    """
    # Get all detections for this frame
    detections = _get_frame_detections(db_client, frame_id)

    # Read evidence trail
    evidence_trail = _read_evidence_trail()

    # Locate the frame image
    viz_filename = f"frame_{frame_id:06d}.jpg"
    image_path = PROJECT_ROOT / "out" / "viz" / viz_filename
    image_base64 = _encode_image_base64(image_path)

    # Generate the HTML ticket
    html = generate_ticket_html(
        frame_id=frame_id,
        detections=detections,
        evidence_trail=evidence_trail,
        image_path=image_path,
        image_base64=image_base64,
    )

    # Write ticket to disk
    tickets_dir = PROJECT_ROOT / "out" / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)
    ticket_file = tickets_dir / f"ticket_{frame_id}.html"
    ticket_file.write_text(html, encoding="utf-8")

    return {
        "ok": True,
        "path": str(ticket_file),
        "filename": ticket_file.name,
        "frame_id": frame_id,
        "num_detections": len(detections),
        "message": f"Ticket created for frame {frame_id} with {len(detections)} detection(s)",
    }

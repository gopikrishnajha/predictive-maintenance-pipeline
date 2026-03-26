#!/usr/bin/env python3
"""
DLStreamer (OpenVINO Execution Provider) inference for YOLO.
Uses Intel DLStreamer via Docker for GPU-accelerated inference pipeline.
Writes results to JSONL file and SQLite database.
Uses gvawatermark for video visualization output.
"""

import subprocess
import json
import yaml
import os
import shutil
from pathlib import Path
import argparse
from scripts.clean_slate import clean_database, clean_outputs

# Import SQLite client
try:
    from src.utility.sqlite_client import SQLiteClient
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("⚠️  SQLite client not available.")


class_names = {
    0: 'Deformation',
    1: 'Obstacle',
    2: 'Rupture',
    3: 'Disconnect',
    4: 'Misalignment',
    5: 'Deposition'
}


def parse_dlstreamer_output(raw_jsonl_path, images_dir):
    """
    Parse DLStreamer JSON array output into per-frame JSONL with source filenames.

    DLStreamer writes a JSON array of frame objects. We map sequential indices
    back to the original sorted image filenames.

    Returns:
        list of frame dicts with 'source' and 'objects' keys
    """
    with open(raw_jsonl_path) as f:
        content = f.read().strip().rstrip(',')
        if not content.startswith('['):
            content = '[' + content + ']'
        data = json.loads(content)

    # Get original sorted filenames (same order as hard links created for multifilesrc)
    images = sorted(Path(images_dir).glob('*.jpg'))

    frames = []
    for idx, frame in enumerate(data):
        source = images[idx].name if idx < len(images) else f'{idx:06d}.jpg'

        # Convert DLStreamer detection format to our standard format
        objects = []
        for obj in frame.get('objects', []):
            det = obj.get('detection', {})
            bbox = det.get('bounding_box', {})

            objects.append({
                'detection': {
                    'label': det.get('label', 'Unknown'),
                    'confidence': det.get('confidence', 0.0),
                    'bounding_box': {
                        'x_min': bbox.get('x_min', 0.0),
                        'y_min': bbox.get('y_min', 0.0),
                        'width': bbox.get('x_max', 0.0) - bbox.get('x_min', 0.0),
                        'height': bbox.get('y_max', 0.0) - bbox.get('y_min', 0.0)
                    }
                }
            })

        frames.append({
            'source': source,
            'objects': objects
        })

    return frames


def parse_dlstreamer_output_video(raw_jsonl_path):
    """
    Parse DLStreamer JSON array output for video input.

    For video mode, there are no original filenames to map to, so we generate
    sequential frame_NNNNNN.jpg names (matching run_inference_ov.py convention).

    Returns:
        list of frame dicts with 'source' and 'objects' keys
    """
    with open(raw_jsonl_path) as f:
        content = f.read().strip().rstrip(',')
        if not content.startswith('['):
            content = '[' + content + ']'
        data = json.loads(content)

    frames = []
    for idx, frame in enumerate(data):
        source = f'frame_{idx:06d}.jpg'

        objects = []
        for obj in frame.get('objects', []):
            det = obj.get('detection', {})
            bbox = det.get('bounding_box', {})

            objects.append({
                'detection': {
                    'label': det.get('label', 'Unknown'),
                    'confidence': det.get('confidence', 0.0),
                    'bounding_box': {
                        'x_min': bbox.get('x_min', 0.0),
                        'y_min': bbox.get('y_min', 0.0),
                        'width': bbox.get('x_max', 0.0) - bbox.get('x_min', 0.0),
                        'height': bbox.get('y_max', 0.0) - bbox.get('y_min', 0.0)
                    }
                }
            })

        frames.append({
            'source': source,
            'objects': objects
        })

    return frames


def run_dlstreamer_inference(images_dir, output_file, model_xml, model_proc,
                             device='GPU', threshold=0.25, img_size=640,
                             nireq=4, num_images=None, video_path=None,
                             inference_interval=1):
    """
    Run DLStreamer inference via Docker.

    Supports two modes:
    - Image directory: Creates sequential hard links for multifilesrc
    - Video file: Uses filesrc ! decodebin with inference-interval and gvawatermark
    """
    workspace_root = Path(__file__).parent.resolve()
    output_path = Path(output_file).resolve()

    # Determine input mode
    use_video = video_path and Path(video_path).exists()
    seq_dir = None
    n_frames = 0

    if use_video:
        video_abs = Path(video_path).resolve()
        # Count frames for reporting
        import cv2
        cap = cv2.VideoCapture(str(video_abs))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_frames = n_total  # Process ALL frames in video mode
        print(f"Processing all {n_frames} frames from video: {video_path}")
        if inference_interval > 1:
            print(f"  inference-interval={inference_interval} (running inference every {inference_interval}th frame)")
    else:
        images_path = Path(images_dir).resolve()
        image_files = sorted(images_path.glob('*.jpg'))
        if not image_files:
            raise RuntimeError(f"No .jpg images found in {images_dir}")
        if num_images:
            image_files = image_files[:num_images]
        n_frames = len(image_files)

        # Create sequential hard links for multifilesrc compatibility
        seq_dir = workspace_root / '.tmp_seq_images'
        if seq_dir.exists():
            shutil.rmtree(seq_dir)
        seq_dir.mkdir(parents=True)

        print(f"Preparing {n_frames} sequential image links...")
        for idx, img_file in enumerate(image_files):
            link_path = seq_dir / f'{idx:06d}.jpg'
            os.link(str(img_file), str(link_path))

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure viz directory exists for gvawatermark output (image mode)
        (workspace_root / 'out' / 'viz').mkdir(parents=True, exist_ok=True)

        # Build Docker command
        docker_output = '/workspace/out/detections_raw.jsonl'

        gpu_args = []
        user_args = []
        if device == 'GPU' and Path('/dev/dri').exists():
            gpu_args = ['--device=/dev/dri:/dev/dri']
            try:
                import grp
                for group_name in ['render', 'video']:
                    try:
                        gid = grp.getgrnam(group_name).gr_gid
                        gpu_args.append(f'--group-add={gid}')
                    except KeyError:
                        pass  # Group does not exist on this system
            except ImportError:
                pass  # grp module not available on this platform
            user_args = ['--user', f'{os.getuid()}:{os.getgid()}']

        model_xml_abs = Path(model_xml).resolve()
        model_proc_abs = Path(model_proc).resolve()
        ov_models_root = (workspace_root / 'models' / 'ov_models').resolve()
        model_det_container = '/workspace/ov_models/' + str(model_xml_abs.relative_to(ov_models_root))
        model_proc_container = '/workspace/ov_models/' + str(model_proc_abs.relative_to(ov_models_root))

        # Build volume mounts and GStreamer pipeline based on input mode
        volumes = [
            '-v', f'{workspace_root}/datasets:/workspace/datasets:ro',
            '-v', f'{workspace_root}/models/ov_models:/workspace/ov_models:ro',
            '-v', f'{workspace_root}/out:/workspace/out',
        ]

        if use_video:
            # Video mode: mount video file, use filesrc ! decodebin
            # gvawatermark overlays detections on video for visualization output
            video_abs = Path(video_path).resolve()
            video_dir = str(video_abs.parent)
            video_name = video_abs.name
            volumes.extend([
                '-v', f'{video_dir}:/workspace/video:ro',
            ])
            
            # inference-interval: run detection every Nth frame
            inference_interval_prop = f' inference-interval={inference_interval}' if inference_interval > 1 else ''
            
            # Output both annotated video AND individual frame images (for SQL frame_id lookup)
            annotated_video_output = '/workspace/out/annotated_output.mp4'
            viz_frames_pattern = '/workspace/out/viz/frame_%06d.jpg'
            
            gst_pipeline = (
                f'gst-launch-1.0 -e '
                f'filesrc location=/workspace/video/{video_name} ! decodebin ! videoconvert ! '
                f'video/x-raw,format=BGRx ! '
                f'gvadetect model={model_det_container} model-proc={model_proc_container} device={device} nireq={nireq} threshold={threshold}{inference_interval_prop} ! '
                f'gvametaconvert format=json add-empty-results=true ! '
                f'gvametapublish method=file file-format=json file-path={docker_output} ! '
                f'gvawatermark ! videoconvert ! video/x-raw,format=RGB ! '
                f'tee name=t '
                f't. ! queue ! jpegenc ! multifilesink location={viz_frames_pattern} '
                f't. ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=4000 ! mp4mux ! '
                f'filesink location={annotated_video_output}'
            )
        else:
            # Image mode: mount seq dir and use multifilesrc
            # gvawatermark overlays detections, multifilesink saves annotated images
            image_pattern = '/workspace/.tmp_seq_images/%06d.jpg'
            viz_output_pattern = '/workspace/out/viz/frame_%06d.jpg'
            volumes.extend(['-v', f'{seq_dir}:/workspace/.tmp_seq_images:ro'])
            
            gst_pipeline = (
                f'gst-launch-1.0 -e '
                f'multifilesrc location={image_pattern} index=0 start-index=0 stop-index=$((N-1)) num-buffers=$N ! '
                f'jpegdec ! videoconvert ! videoscale ! '
                f'video/x-raw,format=BGRx,width=$IMG_SIZE,height=$IMG_SIZE ! '
                f'gvadetect model=$MODEL_DET model-proc=$MODEL_PROC device=$OV_DEVICE nireq=$NIREQ threshold=$THRESHOLD ! '
                f'gvametaconvert format=json add-empty-results=true ! '
                f'gvametapublish method=file file-format=json file-path={docker_output} ! '
                f'gvawatermark ! videoconvert ! video/x-raw,format=RGB ! '
                f'jpegenc ! multifilesink location={viz_output_pattern}'
            )

        # Environment variables (used by image mode's shell expansion)
        env_args = [
            '-e', f'IMAGE_PATTERN=/workspace/.tmp_seq_images/%06d.jpg',
            '-e', f'MODEL_DET={model_det_container}',
            '-e', f'MODEL_PROC={model_proc_container}',
            '-e', f'OV_DEVICE={device}',
            '-e', f'IMG_SIZE={img_size}',
            '-e', f'N={n_frames}',
            '-e', f'THRESHOLD={threshold}',
            '-e', f'NIREQ={nireq}',
        ]

        docker_cmd = [
            'docker', 'run', '--rm', '-i', '--network', 'host',
            *gpu_args,
            *user_args,
            *env_args,
            *volumes,
            'intel/dlstreamer:latest',
            'bash', '-lc', gst_pipeline
        ]

        print(f"Running DLStreamer inference on {n_frames} {'video frames' if use_video else 'images'} (device: {device})...")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            stderr = result.stderr
            # Filter out harmless GStreamer plugin warnings
            error_lines = [l for l in stderr.split('\n')
                           if l.strip() and 'GStreamer-WARNING' not in l and 'libva info' not in l]
            if error_lines:
                print(f"DLStreamer stderr:\n{''.join(error_lines[:10])}")
            raise RuntimeError(f"DLStreamer inference failed with exit code {result.returncode}")

        print("✓ DLStreamer inference completed")

        # The raw output is at out/detections_raw.jsonl
        raw_output = workspace_root / 'out' / 'detections_raw.jsonl'
        return str(raw_output), n_frames

    finally:
        # Cleanup temp sequential links
        if seq_dir and seq_dir.exists():
            shutil.rmtree(seq_dir)


def run_inference(model_path, model_proc_path, images_dir, output_file,
                  device='GPU', conf_threshold=0.25, num_images=None,
                  nireq=4, sqlite_db_path=None, clear_on_run=True,
                  video_path=None, inference_interval=1):
    """Run full inference pipeline: DLStreamer → parse → JSONL + SQLite."""

    # Initialize SQLite client
    sqlite_client = None
    if SQLITE_AVAILABLE and sqlite_db_path:
        try:
            sqlite_client = SQLiteClient(db_path=sqlite_db_path)
            print(f"✓ SQLite client connected to {sqlite_db_path}")
        except Exception as e:
            print(f"⚠️  Failed to connect to SQLite: {e}")
            print("   Continuing with JSONL output only...")

    # Run DLStreamer inference
    raw_output, n_processed = run_dlstreamer_inference(
        images_dir=images_dir,
        output_file=output_file,
        model_xml=model_path,
        model_proc=model_proc_path,
        device=device,
        threshold=conf_threshold,
        num_images=num_images,
        nireq=nireq,
        video_path=video_path,
        inference_interval=inference_interval
    )

    # Parse DLStreamer output into standard format
    use_video = video_path and Path(video_path).exists()
    print("Parsing DLStreamer output...")

    if use_video:
        # For video input, generate frame_NNNNNN.jpg names
        frames = parse_dlstreamer_output_video(raw_output)
    else:
        frames = parse_dlstreamer_output(raw_output, images_dir)

    # Get original image dimensions for SQLite pixel coordinates
    import cv2
    if use_video:
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        if ret:
            video_h, video_w = first_frame.shape[:2]
        else:
            video_h, video_w = 640, 640
        cap.release()
        image_files = []  # No image files in video mode
    else:
        images_path = Path(images_dir)
        image_files = sorted(images_path.glob('*.jpg'))
        if num_images:
            image_files = image_files[:num_images]
        video_h, video_w = 640, 640  # Not used in image mode

    # Write JSONL output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    backend_detections = []

    with open(output_path, 'w') as f:
        for frame_idx, frame in enumerate(frames):
            f.write(json.dumps(frame) + '\n')
            total_detections += len(frame['objects'])

            # Prepare detections for SQLite
            if sqlite_client and frame['objects']:
                # Get original image dimensions
                if use_video:
                    orig_h, orig_w = video_h, video_w
                elif frame_idx < len(image_files):
                    img = cv2.imread(str(image_files[frame_idx]))
                    if img is not None:
                        orig_h, orig_w = img.shape[:2]
                    else:
                        orig_h, orig_w = 640, 640
                else:
                    orig_h, orig_w = 640, 640

                for det in frame['objects']:
                    bbox = det['detection']['bounding_box']
                    x = bbox['x_min'] * orig_w
                    y = bbox['y_min'] * orig_h
                    width = bbox['width'] * orig_w
                    height = bbox['height'] * orig_h

                    backend_detections.append({
                        'frame_id': frame_idx,
                        'label': det['detection']['label'],
                        'confidence': det['detection']['confidence'],
                        'x': int(x),
                        'y': int(y),
                        'width': int(width),
                        'height': int(height)
                    })

    # Clean up raw DLStreamer output
    raw_path = Path(raw_output)
    if raw_path.exists():
        raw_path.unlink()

    # Write to SQLite database
    if sqlite_client and backend_detections:
        try:
            if clear_on_run:
                print()
                clean_database(db_path=sqlite_db_path)
            else:
                print(f"\nAppending to existing SQLite data (clear_on_run: false)...")

            print(f"Writing {len(backend_detections)} detections to SQLite...")
            sqlite_client.insert_detections_batch(backend_detections)
            print(f"✓ Successfully wrote detections to SQLite ({sqlite_db_path})")
        except Exception as e:
            print(f"⚠️  Failed to write to SQLite: {e}")
        finally:
            sqlite_client.close()

    print(f"\n{'='*60}")
    print(f"Inference Complete (DLStreamer/OEP)")
    print(f"{'='*60}")
    print(f"Frames processed: {len(frames)}")
    print(f"Total detections: {total_detections}")
    print(f"Average per frame: {total_detections/max(len(frames),1):.1f}")
    print(f"JSONL output: {output_path}")
    if sqlite_client:
        print(f"SQLite output: {sqlite_db_path}")
    if use_video:
        annotated = Path('out/annotated_output.mp4')
        if annotated.exists():
            print(f"Annotated video: {annotated}")
        viz_dir = Path('out/viz')
        if viz_dir.exists():
            viz_count = len(list(viz_dir.glob('frame_*.jpg')))
            if viz_count:
                print(f"Annotated frames: {viz_dir}/ ({viz_count} images, frame_NNNNNN.jpg matches SQL frame_id)")
    else:
        viz_dir = Path('out/viz')
        if viz_dir.exists():
            viz_count = len(list(viz_dir.glob('*.jpg')))
            if viz_count:
                print(f"Annotated images: {viz_dir}/ ({viz_count} images)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLStreamer (OEP) inference for YOLO with SQLite output')
    parser.add_argument('--model', default=None,
                        help='Path to OpenVINO model XML (default: from config)')
    parser.add_argument('--model-proc', default=None,
                        help='Path to model_proc.json (default: alongside model)')
    parser.add_argument('--images', default='datasets/pipeline_defects_detection/images/val',
                        help='Directory containing images')
    parser.add_argument('--video', default=None,
                        help='Path to input video file (overrides --images)')
    parser.add_argument('--output', default='out/detections.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--device', default='GPU', choices=['CPU', 'GPU'],
                        help='Inference device')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Limit number of images to process (image mode only)')
    parser.add_argument('--inference-interval', type=int, default=1,
                        help='Run inference every Nth frame (video mode, DLStreamer native, default: 1)')
    parser.add_argument('--nireq', type=int, default=4,
                        help='Number of inference requests (default: 4)')
    parser.add_argument('--config', default=None,
                        help='Path to config file (default: reads from config.json)')

    args = parser.parse_args()

    # Load config
    sqlite_db_path = 'out/sql_data/detections.db'
    clear_on_run = True
    clear_outputs = True
    use_case_id = 'pipeline_defects_detection'

    if args.config:
        config_path = args.config
    else:
        # Read use-case-id from config.json
        with open('config.json', 'r') as f:
            main_config = json.load(f)
        use_case_id = main_config.get('use-case-id', 'pipeline_defects_detection')
        config_path = f'config/{use_case_id}.yaml'

    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Load SQLite config
            sqlite_cfg = config.get('sqlite', {})
            sqlite_db_path = sqlite_cfg.get('db_path', 'out/sql_data/detections.db')
            clear_on_run = sqlite_cfg.get('clear_on_run', True)
            clear_outputs = sqlite_cfg.get('clear_outputs', True)

            # Load model path from config if not provided via --model
            if not args.model:
                inference_cfg = config.get('inference', {})
                args.model = inference_cfg.get('model_path', f'models/ov_models/{use_case_id}/best.xml')
            
            # Load images path from config if user didn't override via --images
            inference_cfg = config.get('inference', {})
            config_images_path = inference_cfg.get('images_path')
            if config_images_path and args.images == 'datasets/pipeline_defects_detection/images/val':
                args.images = config_images_path
            
            # Load video path from config if not provided via --video
            if not args.video:
                input_mode = inference_cfg.get('input_mode', 'images')
                if input_mode == 'video':
                    args.video = inference_cfg.get('video_path')
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            print("   Continuing with JSONL output only...")

    # Final fallback if model still not set
    if not args.model:
        args.model = f'models/ov_models/{use_case_id}/best.xml'

    # Default model_proc path (alongside model XML)
    if not args.model_proc:
        args.model_proc = str(Path(args.model).parent / 'model_proc.json')

    # Clean outputs directory if specified in config
    if clear_outputs:
        clean_outputs(out_dir='out')
        print()
    else:
        print(f"📌 Keeping existing outputs (clear_outputs: false)\n")
        Path('out').mkdir(parents=True, exist_ok=True)

    run_inference(args.model, args.model_proc, args.images, args.output,
                  args.device, args.conf_threshold, args.num_images,
                  args.nireq, sqlite_db_path, clear_on_run, args.video,
                  args.inference_interval)

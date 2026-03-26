# Predictive Maintenance Pipeline — Quick Start

Get started with predictive_maintenance_pipeline step by step.

## Prerequisites
- Intel CPU with iGPU (12th Gen+)
- Linux OS
- Conda (Anaconda or Miniconda)
- **No Docker required** (SQLite is embedded)

---

## 1. Create Environment

Run the setup script to create the conda environment and verify devices:

```bash
bash setup/setup.sh
```

This will:
- Check prerequisites (conda)
- Create and configure the `pace` conda environment
- Verify available OpenVINO devices (CPU, GPU, NPU)
- Create data and output directories

---

## 2. Download Data

> **⚠️ DISCLAIMER:** You are solely responsible for ensuring you have the necessary rights, permissions, and licenses to download and use any dataset. We take no responsibility for any misuse of data or violation of terms of service.

Download and prepare a dataset by providing the URL as a command-line argument:

```bash
conda activate pace
python scripts/download_and_prep_data.py "<dataset_url>"
```

For example:
```bash
python scripts/download_and_prep_data.py "https://www.kaggle.com/api/v1/datasets/download/simplexitypipeline/pipeline-defect-dataset"
```

Optional arguments:
```bash
# Custom train/val split ratio (default: 90/10)
python scripts/download_and_prep_data.py "<dataset_url>" --train-ratio 0.8

# Custom output directory
python scripts/download_and_prep_data.py "<dataset_url>" --output datasets/my_dataset

# Specific random seed for reproducibility
python scripts/download_and_prep_data.py "<dataset_url>" --seed 42
```

The script will download, extract, split (train/val), and create a `dataset.yaml` for training.

---

## 3. Training Recipe

### Train a YOLO Model (PyTorch)

Train a YOLOv8 model on your prepared dataset. Adjust `device` based on your available GPUs:

```bash
conda activate pace
yolo detect train \
    data=datasets/pipeline_defects_detection/dataset.yaml \
    model=yolov8s.pt \
    imgsz=640 \
    epochs=50 \
    device=0,1,2,3,4,5
```

> **Note:** Adjust `data=` to point to your `dataset.yaml`, and `device=` to match your GPU setup (e.g. `device=0` for a single GPU, or `device=cpu` for CPU-only training).

### Place the Trained Model

After training completes, copy the best checkpoint to the expected location:

```bash
USE_CASE_ID=$(python3 -c "import json; print(json.load(open('config.json'))['use-case-id'])")
cp runs/detect/train/weights/best.pt models/pt_models/${USE_CASE_ID}.pt
```

### Convert to OpenVINO

Convert the trained PyTorch model to OpenVINO IR format:

```bash
conda activate pace
USE_CASE_ID=$(python3 -c "import json; print(json.load(open('config.json'))['use-case-id'])")
python setup/convert_to_openvino.py \
    --input models/pt_models/${USE_CASE_ID}.pt \
    --output models/ov_models/${USE_CASE_ID}
```

---

## 4. Downloading and Exporting LLMs

The pipeline requires two types of LLMs: one **generic LLM** for agent reasoning and one **SQL model** for text-to-SQL queries.

> **⚠️ DISCLAIMER:** You are solely responsible for ensuring you have the necessary rights, permissions, and licenses to download and use any model. We take no responsibility for any misuse or violation of terms of service.

We recommend:
- **Generic LLM:** `microsoft/Phi-4-mini-instruct`
- **SQL Model:** `defog/sqlcoder-7b-2`

### Configure Your Model Selection

Edit `setup/model_list.txt` and move your chosen models from the `[supported-models]` section to the `[download]` section. You need at least one generic LLM and one SQL model:

```
[download]
microsoft/Phi-4-mini-instruct
defog/sqlcoder-7b-2

[supported-models]
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
meta-llama/Llama-3.2-3B-Instruct
...
```

### Download and Export

```bash
conda activate pace
python scripts/download_these_models.py
```

This reads the `[download]` section from `setup/model_list.txt` and exports each model to OpenVINO format with appropriate quantization. Models already exported will be skipped.

---

## Run Pipeline (Web UI)

Launch the web application for a browser-based interface to run the complete pipeline,
interact with the chat interface, and view ticketing outputs:

```bash
conda activate pace
python scripts/launch_web_app.py
```

This starts a Flask server at `http://localhost:5000` where you can:
- **Run the complete pipeline** — trigger inference and agent orchestration from the browser
- **Interactive chat** — query analysis results and the detections database in natural language
- **View agent outputs** — browse analysis reports, evidence audit trails, and generated tickets
- **Monitor system status** — track pipeline execution progress in real time

---

## Run Pipeline (Command-Line)

### Complete End-to-End (Video Mode — Default)
```bash
conda activate pace
python run_complete_pipeline.py
```

This runs:
- YOLO inference via DLStreamer on video input → SQLite database
- Multi-agent analysis (Policy, Analysis, Evidence)
- Inline visualization via DLStreamer's `gvawatermark`

**Expected output:**
- `out/sql_data/detections.db` - SQLite database with detections
- `out/agent/*.txt` - Agent analysis reports
- `out/agent/*.json` - Structured analysis data
- `out/viz/*.jpg` - Annotated frames with bounding boxes
- `out/annotated_output.mp4` - Annotated video

### Individual Steps
```bash
conda activate pace

# 1. Inference — image mode (writes to SQLite, gvawatermark annotates frames)
python run_inference_oep.py --num-images 100

# 1b. Inference — video mode
python run_inference_oep.py --video datasets/pipeline_defects_detection/video/input.mp4

# 2. Agent analysis (reads from SQLite)
python -m scripts.run_agent_orchestration
```

> **Note:** Visualization is done inline by DLStreamer's `gvawatermark` during inference. Annotated frames are saved to `out/viz/` and annotated video to `out/annotated_output.mp4`.

### Interactive Chat (Command-Line)

For a menu-driven command-line interface to query analysis results, view evidence
audit trails, and run natural-language SQL queries against the detections database:

```bash
conda activate pace
python interactive_chat.py
```

## LLM Server Mode (Optional)

For better performance with shared LLM:

```bash
# Terminal 1: Start LLM server
conda activate pace
bash scripts/run_llm_server.sh

# Terminal 2: Run agents with server
# Edit config/pipeline_defects_detection.yaml: mode: server
python -m scripts.run_agent_orchestration
```

## Clean & Rerun
```bash
conda activate pace
python scripts/clean_slate.py  # Cleans SQLite DB & outputs
python run_complete_pipeline.py
```

## Configuration

Edit `config/pipeline_defects_detection.yaml`:

```yaml
# Use local LLM
glue:
  mode: model
  model_id: models/ov_models/llms/phi-3.5-mini
  device: GPU

# OR use LLM server
glue:
  mode: server
  server_url: http://localhost:8000

# OR use fallback (no LLM)
glue:
  mode: fallback
```

## View Data

```bash
# View SQLite samples
python scripts/read_sqlite_samples.py --limit 10

# Query database
python -c "from src.sqlite_client import SQLiteClient; \
           c = SQLiteClient('out/sql_data/detections.db'); \
           print(c.execute('SELECT label, COUNT(*) FROM detections GROUP BY label'))"
```

---

For detailed documentation, see [README.md](README.md)

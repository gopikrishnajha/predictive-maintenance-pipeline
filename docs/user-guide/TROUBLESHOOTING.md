# Troubleshooting

## SQLite database locked

```bash
lsof out/sql_data/detections.db
pkill -f "python.*run_inference"
python scripts/clean_slate.py
```

## LLM server connection refused

```bash
curl http://localhost:8000/health
bash scripts/run_llm_server.sh
# Or switch to local model: set mode: model in config YAML
```

## OpenVINO GPU device not found

```bash
python -c "import openvino as ov; print(ov.Core().available_devices)"
# If GPU not listed, set device: CPU in config YAML
```

## GPU produces incorrect detections (false positives)

Intel GPU defaults to FP16 which causes quantization errors in YOLO output. The inference code automatically uses FP32 on GPU — no action needed.

## Model conversion fails

```bash
conda activate pace
pip install --upgrade ultralytics openvino openvino-dev
python -c "from ultralytics import YOLO; model = YOLO('models/pt_models/pipeline_defects_detection.pt'); print(model.info())"
```

## No detections in output

```bash
conda activate pace
python run_inference_oep.py --num-images 50 --conf-threshold 0.1
python scripts/read_sqlite_samples.py --limit 5
```

## Agent execution fails

```bash
conda activate pace
ls -lh models/ov_models/llms/   # Check if LLM models exist
python scripts/read_sqlite_samples.py --limit 5  # Check if DB has data
# Try fallback mode: set mode: fallback in config YAML
```

## Out of memory (OOM) when loading LLM

```bash
# Use CPU instead of GPU: set device: CPU in config YAML
# Or use LLM server mode (loads model once):
bash scripts/run_llm_server.sh
# Then set mode: server in config YAML
```

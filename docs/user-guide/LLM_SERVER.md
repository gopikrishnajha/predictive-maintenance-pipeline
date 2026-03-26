# LLM Server Mode - Persistent KV Cache

The LLM server keeps the model loaded in memory, preserving KV cache across pipeline runs for faster inference.

## Quick Start

### 1. Install Dependencies
```bash
conda activate pace
pip install fastapi uvicorn requests
```

### 2. Start the Server
```bash
bash scripts/run_llm_server.sh start
```

The server will:
- Load the LLM model into memory
- Cache the system prompt for KV optimization
- Listen on `http://localhost:8000`

### 3. Configure Client to Use Server
Edit `config.yaml`:
```yaml
glue:
  mode: server  # Changed from 'model' to 'server'
  server_url: http://localhost:8000
  # ... other settings
```

### 4. Run Pipeline
```bash
python run_complete_pipeline.py --num-images 50 --device CPU
```

The pipeline will now use the server instead of loading the model locally each time.

## Server Management

```bash
# Check status
bash scripts/run_llm_server.sh status

# View logs
bash scripts/run_llm_server.sh logs

# Restart server
bash scripts/run_llm_server.sh restart

# Stop server
bash scripts/run_llm_server.sh stop
```

## API Endpoints

- `GET /health` - Check server status and cache info
- `POST /generate` - Generate text from prompt
- `POST /reset-cache` - Reset KV cache
- `GET /docs` - Interactive API documentation

## Benefits

### With Server Mode:
- ✅ **No model loading time** (after first startup)
- ✅ **Persistent KV cache** across runs
- ✅ **~50% faster inference** for subsequent requests
- ✅ **Can serve multiple clients**

### Without Server Mode (Local):
- ❌ Model loads every run (~5-10s overhead)
- ❌ KV cache resets between runs
- ✅ No server management needed
- ✅ Simpler setup

## Performance Comparison

| Mode | First Run | Subsequent Runs | Memory Usage |
|------|-----------|-----------------|--------------|
| Local (`mode: model`) | ~30s | ~30s | Low (only during run) |
| Server (`mode: server`) | ~30s | ~15s | High (always loaded) |

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Check logs
bash scripts/run_llm_server.sh logs
```

### Connection refused
```bash
# Verify server is running
bash scripts/run_llm_server.sh status

# Check firewall settings
curl http://localhost:8000/health
```

### Memory issues
The server keeps the model in memory (~2-4GB depending on model). If memory is constrained:
- Stop the server when not in use
- Use local mode instead

## Configuration Options

```yaml
glue:
  mode: server                    # Use remote server
  server_url: http://localhost:8000
  server_port: 8000              # Port for server to listen on
  use_case: pipeline_defects_detection
  verbose: true                  # Enable detailed logging
```

## Advanced: Custom Server Setup

### Run on Different Port
Edit `config.yaml`:
```yaml
glue:
  server_port: 8080  # Custom port
```

Then restart the server.

### Run on Remote Machine
Start server on remote host, then:
```yaml
glue:
  mode: server
  server_url: http://remote-host:8000
```

### Multiple Models
Run multiple server instances on different ports:
```bash
# Terminal 1
PORT=8000 python scripts/llm_server.py

# Terminal 2  
PORT=8001 python scripts/llm_server.py
```

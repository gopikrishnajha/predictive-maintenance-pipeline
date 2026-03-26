# Predictive Maintenance Pipeline Blueprint

> **Blueprint Series — Edge AI Predictive Maintenance for Critical Infrastructure**
> Multi-Agent Reasoning + Edge Vision with Intel OpenVINO

Industrial infrastructure — pipelines, solar panels, bridges — degrades invisibly.
Corrosion forms inside joints, obstacles accumulate along routes, and micro-fractures
propagate beneath coatings. Traditional inspection is manual, periodic, and reactive:
a human walks the line, sees what is already visible, and files a report. What if an
edge-deployed AI system could detect defects continuously from video or imagery,
reason about their severity through a structured agent pipeline, and produce
auditable reports — all running on Intel hardware at the edge, with no cloud
dependency?

That is the premise of this blueprint. Built around a real pipeline defect
dataset with six defect classes, it demonstrates how edge vision AI and multi-agent
reasoning with Intel OpenVINO can power a new generation of predictive maintenance
workflows that go beyond simple detection into structured analysis, policy
enforcement, and evidence-grade audit trails.

The current implementation demonstrates this on a **pipeline defect detection** use
case, but the architecture is designed for domain portability — it can be extended to
other inspection domains such as solar panel defect detection, bridge structural
assessment, or manufacturing quality control with minimal changes to the
configuration and prompt files. Similarly, this version uses images and video as
input data, but the pipeline architecture is designed to accommodate additional
sensor modalities in future iterations — radar, LiDAR, thermal imaging, and other
Non-Destructive Testing (NDT) data sources — broadening the system's applicability
across industrial predictive maintenance scenarios.

The result is a three-unit architecture: a vision inference layer that detects defects
in real time, a structured data layer that persists every detection in SQLite, and a
multi-agent reasoning layer — coordinated by LangGraph — where specialized agents
generate policy rules, filter and analyze detections, produce compliance audit trails,
and render self-contained HTML tickets. All of it runs on a single Intel edge node.


## The Dataset and Defect Classes

The system is designed around industrial pipeline defect detection, with a dataset
that exercises six distinct defect categories:

| Defect Class   | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| Deformation    | Structural warping or bending of pipeline segments                         |
| Obstacle       | Foreign objects or debris obstructing the pipeline path                    |
| Rupture        | Breaks or tears in the pipeline wall — a critical, high-severity defect    |
| Disconnect     | Separation at joints or coupling points — a critical, high-severity defect |
| Misalignment   | Positional offset between connected pipeline segments                      |
| Deposition     | Material buildup (corrosion, sediment, biological growth) on surfaces      |

The dataset is organized in standard YOLO format with train/validation splits and
supports both image batch processing and video stream inference. Images and
annotations are stored under `datasets/pipeline_defects_detection/`.

The defect class taxonomy is not arbitrary — it reflects a severity hierarchy that
the agent reasoning layer exploits. Rupture and Disconnect are treated as critical
defects with elevated confidence thresholds (0.8), while Obstacle requires moderate
confidence (0.5), and Deformation, Misalignment, and Deposition use standard
thresholds (0.4). This severity mapping is encoded in the policy layer and enforced
automatically during analysis.


## The AI Models

Two distinct model families serve complementary roles in the pipeline.

### Detection Model — YOLOv8 on OpenVINO

A YOLOv8 object detection model is trained on the pipeline defect dataset to detect
all six defect classes. The model is converted to OpenVINO Intermediate Representation
(IR) format for optimized inference on Intel hardware.

- **Input size:** 640×640
- **Quantization:** INT8, FP16, or FP32 (configurable per deployment)
- **Inference targets:** Intel iGPU (preferred), CPU, or NPU
- **Framework:** Intel DL Streamer with OpenVINO Execution Provider
- **Post-processing:** YOLO v8 output converter with aspect-ratio-preserving resize

The model is trained using the standard Ultralytics YOLOv8 training pipeline and
exported to OpenVINO IR via the `setup/convert_to_openvino.py` utility. Model
metadata — class count, stride, batch size, and label mapping — is stored alongside
the model in `models/ov_models/pipeline_defects_detection/`.

### Reasoning Models — LLMs on OpenVINO GenAI

The agent reasoning layer uses large language models for policy generation, analysis
report writing, and evidence trail composition. Three LLM backends are supported:

| Backend    | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| **Local Model** | OpenVINO GenAI pipeline running on-device (GPU, CPU, or NPU). Models include `Phi-4-mini-instruct` and `DeepSeek-R1-Distill-Llama-8B`, quantized to INT4 for edge efficiency. |
| **Remote Server** | Persistent LLM server (`scripts/llm_server.py`) with KV cache optimization for repeated queries. Useful for multi-user or high-throughput scenarios. |
| **Fallback** | Rule-based operation with no LLM dependency. Policy defaults to `config/policy_fallback.json`; analysis and evidence use template-based generation. |

An additional specialized model — `defog/sqlcoder-7b-2` — handles natural-language-to-SQL
translation for the interactive query interface, enabling users to ask questions
about the detections database in plain English.

All LLM models are downloaded and converted to OpenVINO format with INT4 quantization
via `scripts/download_these_models.py`, with model identifiers listed in
`setup/model_list.txt`.


## Architecture: The Three-Unit Stack

The architecture is organized into three cleanly separated units, from data
acquisition to intelligent reasoning:

```
┌─────────────────────────────────────────────────────────┐
│                        Web UI                           │
│ Pipeline Execution · Interactive Chat · Agent Outputs   │
├─────────────────────────────────────────────────────────┤
│                 Unit 3: Agent Reasoning                 │
│                                                         │
│  ┌──────────┐  ┌──────────┐ ┌──────────┐  ┌──────────┐  │
│  │  Policy  │  │ Analysis │ │ Evidence │  │ Ticketing│  │
│  │  Agent   │  │  Agent   │ │  Agent   │  │  Agent   │  │
│  └────┬─────┘  └────┬─────┘ └────┬─────┘  └────┬─────┘  │
│       └─────────────┴────────────┴─────────────┘        │
│                         ▲                               │
│                    Meta-Agent                           │
│                   (Coordinator)                         │
├─────────────────────────────────────────────────────────┤
│              Unit 2: Data / Storage Layer               │
│                                                         │
│              SQLite — detections.db                     │
│    frame_id · label · confidence · bbox (x,y,w,h)       │
├─────────────────────────────────────────────────────────┤
│            Unit 1: Inference / Ingestion                │
│                                                         │
│  DL Streamer → YOLOv8 (OpenVINO) → Detection Extract    │
│                                                         │
│  Video Stream / Image Batch → Per-Frame Detections      │
└─────────────────────────────────────────────────────────┘
                         ▲
                  Intel CPU/iGPU/NPU
```

### Unit Descriptions

#### Unit 1: Inference and Ingestion — DL Streamer + OpenVINO

The inference layer processes video streams or image batches through Intel DL Streamer,
a GStreamer-based pipeline framework for video analytics. Each frame passes through the
YOLOv8 detection model running on OpenVINO, producing per-frame bounding box detections
with class labels and confidence scores. Detections are extracted inline and written
directly to the SQLite database.

Two input modes are supported:

- **Video mode** — Processes an MP4 or RTSP stream, running inference at a configurable
  frame interval (e.g., every 5th frame). Supports inline visualization via
  `gvawatermark` for real-time visual feedback.
- **Image batch mode** — Processes a directory of images, one detection pass per image.
  Suitable for offline analysis of inspection imagery.

The inference layer is deliberately simple: its only job is to detect and persist.
All reasoning happens downstream.

#### Unit 2: Data and Storage — SQLite

SQLite serves as the persistence layer between inference and reasoning. The choice of
SQLite is intentional for edge deployment — it requires no server process, no
network configuration, and no administration. The database resides at
`out/sql_data/pipeline_defects_detection.db` and contains a single `detections` table:

| Column      | Type    | Description                              |
|-------------|---------|------------------------------------------|
| `frame_id`  | INTEGER | Source frame identifier                  |
| `label`     | TEXT    | Defect class name                        |
| `confidence`| REAL    | Detection confidence score (0.0–1.0)     |
| `x`         | REAL    | Bounding box center X coordinate         |
| `y`         | REAL    | Bounding box center Y coordinate         |
| `width`     | REAL    | Bounding box width                       |
| `height`    | REAL    | Bounding box height                      |

Indexes are automatically created on `frame_id`, `label`, and `confidence` for
efficient filtering. The database can be cleared on each run or preserved for
longitudinal analysis, controlled via configuration.

#### Unit 3: Agent Reasoning — LangGraph Multi-Agent Orchestration

The reasoning layer is the analytical core of the system. It implements a
hub-and-spoke multi-agent pattern using LangGraph, where a central **Meta-Agent**
coordinates four specialized worker agents. No agent communicates directly with
another — all data flows through the Meta-Agent's shared state.

The agent pipeline executes in two phases:

**Phase 1 — Policy Generation:**
1. Meta-Agent initializes state and loads resources (database, LLMs, prompts)
2. Policy Agent generates filtering rules — confidence thresholds per defect class,
   minimum bounding box sizes, and global constraints — either via LLM reasoning or
   from the fallback policy file

**Phase 2 — Analysis and Audit:**
1. Meta-Agent retrieves raw detections from SQLite and prepares data packages for
   downstream agents
2. Analysis Agent filters detections using the generated policy, computes per-class
   statistics (counts, confidence distributions, top-N detections), and produces a
   structured analysis report
3. Evidence Agent independently filters detections (consistency check), computes
   policy compliance metrics, and generates a timestamped audit trail
4. Ticketing Agent (optional) renders self-contained HTML tickets for flagged frames,
   embedding detection imagery and metadata

Analysis and Evidence agents can execute sequentially (sharing a single LLM on one
device) or in parallel (each with its own LLM instance on separate devices),
configurable via YAML.


## Data Flow

```
Camera / Image Store
        │
        ▼
DL Streamer + OpenVINO (YOLOv8)
  Detect: Deformation · Obstacle · Rupture · Disconnect · Misalignment · Deposition
        │
        ▼
SQLite Database
  frame_id · label · confidence · bbox
        │
        ▼
Meta-Agent (LangGraph Coordinator)
        │
        ├──→ Policy Agent → Filtering rules (JSON)
        │
        ├──→ Analysis Agent → Statistics + Summary report
        │
        ├──→ Evidence Agent → Compliance audit trail
        │
        └──→ Ticketing Agent → HTML tickets with embedded images
        │
        ▼
Output Artifacts
  out/agent/policy.json
  out/agent/analysis_report.json
  out/agent/analysis_summary.txt
  out/agent/evidence.json
  out/agent/evidence_trail.txt
  out/agent/tickets/TICKET-F{frame_id}.html
```


## Agent Descriptions

All four worker agents operate on structured data — filtered detection records,
policy rules, and computed metrics — not on raw video or images. The Meta-Agent
prepares each agent's input from the shared state, ensuring data isolation and
reproducibility.

### Policy Agent

The Policy Agent establishes the filtering rules that govern downstream analysis.
Given the system prompt describing the defect domain and a policy generation prompt,
it produces a structured JSON policy:

```json
{
  "min_conf_global": 0.4,
  "per_class_thresholds": {
    "Deformation": 0.4,
    "Obstacle": 0.5,
    "Rupture": 0.8,
    "Disconnect": 0.8,
    "Misalignment": 0.4,
    "Deposition": 0.4
  },
  "bbox_min_size": {
    "width": 40,
    "height": 40
  }
}
```

Critical defects (Rupture, Disconnect) receive elevated thresholds to minimize false
positives — a false rupture alert is costly. The bounding box minimum size filter
eliminates spurious micro-detections that often result from image noise or model
artifacts.

In fallback mode (no LLM), the policy is loaded directly from
`config/policy_fallback.json`, ensuring the system operates deterministically even
without LLM availability.

### Analysis Agent

The Analysis Agent consumes the filtered detection set and produces a comprehensive
statistical report:

- **Per-class counts** — how many detections of each defect type survived filtering
- **Confidence distributions** — mean, min, max confidence per class
- **Top-N detections** — the highest-confidence detections across all classes
- **Summary narrative** — a human-readable analysis report (LLM-generated or
  template-based)

Output is saved as both structured JSON (`analysis_report.json`) and readable text
(`analysis_summary.txt`).

### Evidence Agent

The Evidence Agent provides the compliance and audit layer. It independently applies
the same policy filter (as a consistency check against the Analysis Agent), computes
policy compliance metrics, and generates a timestamped audit trail documenting:

- Run timestamp and status
- Policy rules applied and their source
- Detection filtering metrics (total → filtered counts, rejection rates)
- Per-class compliance breakdown
- Traceability chain from raw detection to final disposition

This separation of analysis and evidence is deliberate — it mirrors the
separation of duties principle in audit frameworks. The Analysis Agent answers
"what did we find?" while the Evidence Agent answers "can we prove the process was
followed correctly?"

### Ticketing Agent

The Ticketing Agent generates self-contained HTML tickets for individual frames
flagged during analysis. Each ticket embeds the detection image, bounding box
overlays, defect metadata, and confidence scores — producing a portable,
human-reviewable artifact that requires no database access to interpret. Tickets
are saved to `out/agent/tickets/`.


## Prompt System

Prompts are domain-specific and organized in `prompts/{use_case}.txt` using a
section-based format. Each file contains tagged sections that the prompt loader
parses and distributes to the appropriate agents:

| Section       | Purpose                                                    |
|---------------|------------------------------------------------------------|
| `[SYSTEM]`    | Domain context — defect classes, operating constraints      |
| `[POLICY]`    | Instructions for policy generation — thresholds, rules      |
| `[ANALYSIS]`  | Instructions for report generation — structure, format      |
| `[EVIDENCE]`  | Instructions for audit trail generation — compliance format |

Custom sections can be added for domain-specific extensions. The prompt loader
(`src/utility/prompt_loader.py`) parses these sections and makes them available
by lowercase key, allowing agents to retrieve their specific instructions without
knowledge of other agents' prompts.

Two prompt files are included:
- `prompts/pipeline_defects_detection.txt` — Pipeline infrastructure defects
- `prompts/solar_panel_defects_detection.txt` — Solar panel defects (demonstrates
  domain portability)


## Configuration

The system uses a two-level configuration hierarchy.

### Global Config — `config.json`

Selects the active use case:

```json
{
  "use-case-id": "pipeline_defects_detection"
}
```

### Use-Case Config — `config/pipeline_defects_detection.yaml`

Contains all parameters for a specific use case, organized into sections:

**Inference configuration:**
- Model path, device target (GPU/CPU/NPU), confidence and IoU thresholds
- Input mode (video or images), source paths, frame interval

**SQLite configuration:**
- Database path, clear-on-run behavior

**Agent configuration:**
- Execution mode (sequential or parallel)
- Shared device for sequential mode
- Per-agent LLM settings: mode (model/server/fallback), model ID, device
- Analysis parameters: confidence threshold, top-N report count

**LLM global settings:**
- Default mode, model, device
- Max reasoning steps, verbosity, caching, thinking suppression
- Server URL and port (for remote mode)

**SQL query settings:**
- Text-to-SQL model ID and device target


## Architectural Principles

### Scene Data, Not Video

Every agent in the reasoning layer operates on structured detection records — not
video frames, not pixel data. The inference pipeline's output is a stream of
bounding boxes with labels and confidence scores; the SQLite layer persists them as
queryable records; the agents reason over tabular data. This separation has
critical consequences:

- **Privacy by architecture** — no raw video is retained or processed in the reasoning
  layer. Only derived detection metadata flows downstream.
- **Scalability** — detection records are orders of magnitude smaller than video.
  Agents can reason over thousands of frames in seconds.
- **Reproducibility** — the SQLite database is a complete, deterministic record of
  every detection. Re-running agents produces identical results without
  re-running inference.

### Edge-Native Deployment

All inference, LLM reasoning, and data storage runs on-premise on Intel edge
hardware. No data leaves the device. The system requires no cloud connectivity,
no external API calls, and no network dependency beyond initial model download.
This is consistent with operational technology (OT) security requirements in
industrial environments where pipeline infrastructure is monitored.

### LLM Flexibility with Deterministic Fallback

The three-tier LLM backend (local model → remote server → rule-based fallback)
ensures the system degrades gracefully. If the LLM is unavailable or produces
invalid output, the fallback policy and template-based generation maintain
operational continuity. The system never stops working because an LLM is
unreachable.

### Hub-and-Spoke Agent Isolation

No agent communicates directly with another. All data flows through the Meta-Agent's
shared state, with strict namespace isolation — each agent reads only from its
designated input namespace and writes only to its designated output namespace. This
design prevents cascading failures, enables independent agent testing, and makes
the execution order explicit and auditable.


## Setup and Installation

### Prerequisites

- Intel CPU with iGPU (12th Gen or later) or NPU
- Linux (Ubuntu 20.04+)
- Conda (Anaconda or Miniconda)
- Minimum 8 GB RAM; 16 GB recommended for LLM loading

### Step 1 — Environment Setup

```bash
cd /path/to/predictive-maintenance-pipeline
bash setup/setup.sh
```

This creates a dedicated conda environment with Python 3.10, installs all
dependencies from `setup/requirements.txt`, verifies OpenVINO device availability
(CPU, GPU, NPU), and creates required data and output directories.

### Step 2 — Activate Environment

```bash
source setup/activate_env.sh
```

### Step 3 — Download and Prepare Dataset

```bash
python scripts/download_and_prep_data.py "<dataset_url>"
```

Optional parameters: `--train-ratio`, `--output`, `--seed` for custom
train/validation splits.

### Step 4 — Convert Detection Model to OpenVINO

```bash
python setup/convert_to_openvino.py \
    --input models/pt_models/pipeline_defects_detection.pt \
    --output models/ov_models/pipeline_defects_detection
```

### Step 5 — Download and Convert LLM Models

Edit `setup/model_list.txt` to select models, then:

```bash
python scripts/download_these_models.py
```

Models are exported to OpenVINO format with INT4 quantization automatically.


## Running the Pipeline

### Complete End-to-End Pipeline

```bash
python run_complete_pipeline.py --num-images 100 --device GPU
```

Runs YOLO inference on images, persists detections to SQLite, executes all agents
(Policy → Analysis → Evidence), and generates output artifacts.

### Video Mode

```bash
python run_complete_pipeline.py \
    --video datasets/pipeline_defects_detection/video/input.mp4 \
    --device GPU \
    --inference-interval 5
```

Processes every 5th frame with inline `gvawatermark` visualization.

### Inference Only

```bash
python run_inference_oep.py --num-images 100 --device GPU
```

Runs detection inference without agent reasoning — useful for populating the
database before independent agent runs.

### Agent Orchestration Only

```bash
python -m scripts.run_agent_orchestration --use-case pipeline_defects_detection
```

Reads existing detections from SQLite and runs the full agent pipeline. Useful for
re-analyzing data with different policy configurations without re-running inference.

### Interactive Chat

```bash
python interactive_chat.py
```

Menu-driven command-line interface for querying analysis results, viewing evidence
audit trails, running natural-language SQL queries against the detections database,
and exploring cached agent responses.

### Web Application

```bash
python scripts/launch_web_app.py
```

Opens a web interface at `http://localhost:5000` with real-time pipeline execution
streaming, interactive chat, agent output viewing and system status monitoring.


## Output Artifacts

After a complete pipeline run, the following artifacts are produced:

```
out/
├── sql_data/
│   └── pipeline_defects_detection.db        # SQLite detection database
└── agent/
    ├── policy.json                          # Generated policy rules
    ├── analysis_report.json                 # Structured analysis data
    ├── analysis_summary.txt                 # Human-readable analysis report
    ├── evidence.json                        # Structured audit record
    ├── evidence_trail.txt                   # Human-readable audit trail
    └── tickets/
        └── TICKET-F{frame_id}.html          # Per-frame HTML tickets
```

Each artifact is self-contained and interpretable independently. The JSON files
support programmatic consumption; the text files support human review; the HTML
tickets support offline inspection workflows.


## From Blueprint to Deployment

This blueprint represents a complete, end-to-end reference architecture. The key
integration points for deployment practitioners are:

| Component               | Technology                         | Integration Notes                                          |
|-------------------------|------------------------------------|------------------------------------------------------------|
| Video / Image Ingest    | Intel DL Streamer                  | GStreamer pipeline with RTSP, USB, or file sources          |
| Object Detection        | YOLOv8 on OpenVINO                 | Fine-tuned on site-specific defect dataset                 |
| Data Persistence        | SQLite                             | Embedded, serverless — no infrastructure overhead           |
| Agent Orchestration     | LangGraph (hub-and-spoke)          | Meta-Agent coordinates Policy, Analysis, Evidence, Ticketing|
| LLM Reasoning           | OpenVINO GenAI                     | Phi-4-mini-instruct or DeepSeek-R1, INT4 quantized          |
| Text-to-SQL             | sqlcoder-7b-2 on OpenVINO          | Natural language queries against detection database          |
| Interactive Interface   | Flask Web App / CLI                | Real-time streaming, chat, and agent output viewing         |
| Hardware Platform       | Intel 12th Gen+ with iGPU/NPU     | CPU + GPU + NPU; OpenVINO targets all three accelerators    |
| Operating System        | Linux (Ubuntu 20.04+)              | Conda-based environment with full dependency isolation       |

### Extending to New Domains

The architecture is domain-portable. To adapt the blueprint to a new inspection
domain (e.g., solar panels, bridge structures, manufacturing quality):

1. **Train a new detection model** on the domain-specific dataset
2. **Create a new prompt file** in `prompts/` with domain-specific defect classes,
   severity mappings, and analysis instructions
3. **Create a new YAML config** in `config/` with model paths and agent parameters
4. **Update `config.json`** to point to the new use case ID

No agent code changes are required. The prompt system and configuration hierarchy
are designed for this kind of domain swapping.

### Extending to New Sensor Modalities

The current version of the pipeline ingests images and video as its primary data
sources. However, the three-unit architecture — with its clean separation between
ingestion, storage, and reasoning — is designed to accommodate additional sensor
modalities in future iterations. Potential extensions include:

- **Radar and LiDAR** — Subsurface and geometric defect detection for buried or
  occluded infrastructure
- **Thermal imaging** — Heat signature anomalies indicating insulation failure,
  leaks, or electrical faults
- **Acoustic emission sensors** — Vibration and stress-wave analysis for fatigue
  crack detection
- **Ultrasonic testing** — Wall thickness measurement and internal corrosion mapping

Each new modality would feed into the SQLite persistence layer through a
modality-specific ingestion adapter, while the agent reasoning layer remains
unchanged — policy, analysis, and evidence agents operate on structured detection
records regardless of the source sensor.


## Conclusion

The Predictive Maintenance Pipeline blueprint demonstrates that the convergence of
edge vision AI, multi-agent reasoning, and heterogeneous Intel hardware makes it
possible to build industrial inspection systems that are simultaneously more
capable, more auditable, and more deployable than traditional manual workflows.

By separating detection from reasoning, persisting every observation in a queryable
database, and orchestrating specialized agents through a structured hub-and-spoke
pattern, the architecture delivers defect detection, policy enforcement, statistical
analysis, and compliance audit trails from a single unified pipeline. While the
current implementation focuses on pipeline defect detection using image and video
input, the same foundation scales to solar panels, bridges, manufacturing lines, and
any domain where visual — or eventually multi-modal — defect detection meets
structured analytical reasoning.

The system is not a dashboard — it is an operational edge AI pipeline. And with
Intel OpenVINO as its inference and reasoning runtime, it is deployable on real
hardware, with real data, at the edge.

---

*This blueprint is a proof of concept and is not intended for production use. Other
names and brands may be claimed as the property of others.*

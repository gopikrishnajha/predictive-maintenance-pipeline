# Predictive Maintenance Pipeline — Agent Architecture

## Two-Phase Hub-and-Spoke Pattern

The predictive_maintenance_pipeline agent system uses LangGraph with a two-phase hub-and-spoke architecture where all inter-agent communication flows through meta nodes.

### Architecture Diagram

```
                    ┌─────────────────────┐
                    │    META AGENT       │
                    │  (Coordinator)      │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐        ┌──────────┐
   │ POLICY  │         │ ANALYSIS │        │ EVIDENCE │
   │ AGENT   │         │  AGENT   │        │  AGENT   │
   └─────────┘         └──────────┘        └──────────┘
```

**Execution Flow (Sequential):**

```
1. meta_entry (Meta: Phase 1 start)
       │
       ▼
2. policy_fn (Policy Agent: filter detections)
       │
       ▼
3. meta_collect_policy (Meta: Phase 1→2 transition)
       │
       ▼
4. analysis_fn (Analysis Agent: generate report)
       │
       ▼
5. evidence_fn (Evidence Agent: create audit)
       │
       ▼
6. collect_results (Meta: finalize)
       │
       ▼
     END
```

**Meta Agent has 3 functions:**
- `meta_entry` - Initializes workflow
- `meta_collect_policy` - Collects Phase 1 results, prepares Phase 2
- `collect_results` - Finalizes and aggregates all results

### Key Principles

1. **Hub-and-Spoke: 1 Meta Agent + 3 Worker Agents**
   - **Meta Agent**: Coordinator (3 functions: entry, collect_policy, collect_results)
   - **Policy Agent**: Filters detections
   - **Analysis Agent**: Generates reports
   - **Evidence Agent**: Creates audit trails

2. **No Direct Agent-to-Agent Communication**
   - Worker agents only read from `meta` state
   - Worker agents write to their own state fields
   - Only meta can read all agent outputs

3. **Sequential Execution**
   - Meta → Policy → Meta → Analysis → Evidence → Meta → END
   - Avoids OpenVINO GenAI concurrent request bug
   - Maintains clean separation of concerns

### Data Flow

**Phase 1:**
```
VDMS → policy_fn → state.detections_kept
                 → state.policy_metrics
```

**Meta Collection:**
```
state.detections_kept → meta.detections_for_downstream
```

**Phase 2:**
```
meta.detections_for_downstream → analysis_fn → state.analysis_report
                                              → state.analysis_summary

meta.detections_for_downstream → evidence_fn → state.evidence
state.policy_metrics                         → state.evidence_trail
```

### Agent Independence

Each agent:
- ✅ Can use LLM (via `state.llm`)
- ✅ Can query VDMS (via `state.vdms_ops`)
- ✅ Reads only from `meta` or its inputs
- ❌ Cannot directly read other agents' outputs
- ❌ Cannot directly call other agents

### Implementation Files

- **Graph Definition:** [`src/agents/meta_agent.py`](src/agents/meta_agent.py)
- **State Schema:** [`src/agents/state.py`](src/agents/state.py)
- **Policy Agent:** [`src/agents/policy_metrics.py`](src/agents/policy_metrics.py)
- **Analysis Agent:** [`src/agents/analysis_reporting.py`](src/agents/analysis_reporting.py)
- **Evidence Agent:** [`src/agents/evidence_audit.py`](src/agents/evidence_audit.py)
- **LLM Backend:** [`src/agents/openvino_llm.py`](src/agents/openvino_llm.py)

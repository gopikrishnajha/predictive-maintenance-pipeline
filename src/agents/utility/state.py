"""
State definitions for LangGraph hub-and-spoke agent workflow.

In this architecture:
- All shared resources and agent outputs are stored in the 'meta' namespace
- Agents only read from and write to 'meta', never directly from each other
- Meta nodes control data flow between agents
"""

from typing import TypedDict, Dict, Any, Annotated


def merge_meta(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two meta dictionaries for parallel writes.
    Deep merges nested dictionaries.
    """
    if not right:
        return left
    if not left:
        return right
    
    merged = left.copy()
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Deep merge nested dicts
            merged[key] = {**merged[key], **value}
        else:
            # Overwrite for non-dict values
            merged[key] = value
    return merged


class AgentState(TypedDict, total=False):
    """
    Shared state for LangGraph hub-and-spoke workflow.
    
    Structure:
        meta: {
            # Shared resources (available to all agents)
            resources: {
                db_backend: DatabaseBackend instance
                llm: LLM instance (optional)
                config: Configuration dict
                prompts: {
                    policy_prompt: str
                    analysis_prompt: str
                    evidence_prompt: str
                }
            }
            
            # Workflow control
            phase: int (1 or 2)
            started: bool
            completed: bool
            
            # Policy agent namespace
            policy: {
                metrics: {...}
                detections_kept: [...]
                raw_detections: [...]
            }
            
            # Data prepared by meta for downstream agents
            for_analysis: {
                detections: [...]
            }
            for_evidence: {
                policy_metrics: {...}
            }
            
            # Analysis agent namespace
            analysis: {
                report: {...}
                summary: {...}
                summary_text: str
            }
            
            # Evidence agent namespace
            evidence: {
                trail: str
                record: {...}
            }
            
            # Final status
            final_status: {...}
        }
    """
    
    # Use Annotated with reducer for parallel writes to meta
    meta: Annotated[Dict[str, Any], merge_meta]

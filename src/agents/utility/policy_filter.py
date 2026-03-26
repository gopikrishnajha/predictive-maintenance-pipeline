"""
Policy-based detection filtering utility.

Shared by Analysis and Evidence agents to apply policy rules consistently.
"""

from typing import List, Dict, Any


def filter_detections(detections: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter detections based on policy rules.
    
    Args:
        detections: List of detection dicts with structure:
            {
                "label": str,
                "confidence": float,
                "bbox": {"x": float, "y": float, "width": float, "height": float},
                ...
            }
        policy: Policy dict with structure:
            {
                "min_conf_global": float,
                "per_class_thresholds": {"label": float, ...},
                "bbox_min_size": {"width": float, "height": float}
            }
            
    Returns:
        Filtered list of detections that pass policy criteria
    """
    if not detections or not policy:
        return []
    
    min_conf_global = policy.get("min_conf_global", 0.0)
    per_class_thresholds = policy.get("per_class_thresholds", {})
    bbox_min_size = policy.get("bbox_min_size", {"width": 0, "height": 0})
    
    filtered = []
    
    for det in detections:
        label = det.get("label", "")
        confidence = det.get("confidence", 0)
        bbox = det.get("bbox", {})
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)
        
        # Determine threshold for this label
        threshold = per_class_thresholds.get(label, min_conf_global)
        
        # Apply filters
        if (confidence >= threshold and 
            width >= bbox_min_size.get("width", 0) and 
            height >= bbox_min_size.get("height", 0)):
            filtered.append(det)
    
    return filtered

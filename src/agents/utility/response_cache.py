"""
Response cache for LLM - stores and reuses responses to similar prompts.
Uses prompt hashing for exact matches and optional fuzzy matching.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ResponseCache:
    """Cache LLM responses to avoid redundant generation."""
    
    def __init__(self, cache_file: str = "out/llm_cache.json", enabled: bool = True):
        self.cache_file = Path(cache_file)
        self.enabled = enabled
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        
        if self.enabled:
            self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"[ResponseCache] 📦 Loaded {len(self.cache)} cached responses")
            except Exception as e:
                print(f"[ResponseCache] ⚠️ Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[ResponseCache] ⚠️ Failed to save cache: {e}")
    
    def _hash_prompt(self, prompt: str, system_prompt: str = "") -> str:
        """Generate hash for prompt (for exact matching)."""
        combined = system_prompt + "\n\n" + prompt
        # Normalize timestamps and other dynamic values for better cache hits
        normalized = self._normalize_prompt(combined)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt by removing/replacing dynamic values."""
        # Replace timestamps with placeholder
        # Matches: "ts": 1234567890.123, "timestamp": "2026-01-21 10:30:45"
        prompt = re.sub(r'"ts":\s*\d+\.?\d*', '"ts": <TIMESTAMP>', prompt)
        prompt = re.sub(r'"timestamp":\s*"[^"]*"', '"timestamp": "<TIMESTAMP>"', prompt)
        prompt = re.sub(r'Timestamp:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', 'Timestamp: <NORMALIZED>', prompt)
        
        # Normalize "created"/"last_used" timestamps
        prompt = re.sub(r'"(created|last_used)":\s*"[^"]*"', r'"\1": "<TIMESTAMP>"', prompt)
        
        return prompt
    
    def get(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Get cached response if available."""
        if not self.enabled:
            return None
        
        key = self._hash_prompt(prompt, system_prompt)
        
        if key in self.cache:
            self.hits += 1
            entry = self.cache[key]
            entry["last_used"] = datetime.now().isoformat()
            entry["use_count"] = entry.get("use_count", 1) + 1
            return entry["response"]
        
        self.misses += 1
        return None
    
    def put(self, prompt: str, response: str, system_prompt: str = ""):
        """Store response in cache."""
        if not self.enabled:
            return
        
        key = self._hash_prompt(prompt, system_prompt)
        
        self.cache[key] = {
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response": response,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "use_count": 1,
            "response_length": len(response)
        }
        
        # Save after each write
        self._save_cache()
    
    def clear(self):
        """Clear all cached responses."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print(f"[ResponseCache] 🗑️ Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "total_entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_file": str(self.cache_file)
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.stats()
        print(f"\n[ResponseCache] 📊 Statistics:")
        print(f"  Entries: {stats['total_entries']}")
        print(f"  Hits: {stats['hits']} | Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']}")

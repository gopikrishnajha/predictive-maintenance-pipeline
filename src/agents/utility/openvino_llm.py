"""
OpenVINO LLM wrapper for LangGraph integration.
Provides a simple interface to OpenVINO GenAI LLMPipeline.
"""

from typing import Any, Optional, Dict
import openvino_genai as ov_genai
import os
import json
import requests
from .response_cache import ResponseCache


class OpenVINOLLM:
    """OpenVINO GenAI LLM wrapper - simplified for stability."""
    
    def __init__(self, model_path: str, device: str = "CPU", verbose: bool = False, enable_cache: bool = False, suppress_thinking: bool = True):
        self.model_path = model_path
        self.device = device
        self.verbose = verbose
        self.suppress_thinking = suppress_thinking
        self.pipeline = None
        self._model_loaded = False
        self.response_cache = ResponseCache(enabled=enable_cache)  # Response-level caching
    
    def set_cache_prefix(self, prefix: str):
        """Set a common system prompt prefix (for compatibility - not used with response cache)."""
        pass
    
    def start_fresh(self):
        """Reset cache state (for compatibility)."""
        pass
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove thinking/reasoning content from model output.
        
        Args:
            text: Raw model output that may contain thinking content
        
        Returns:
            Cleaned text with thinking content removed
        """
        import re
        
        # Primary method: Remove everything before </think> tag (DeepSeek-R1 format)
        if '</think>' in text:
            parts = text.split('</think>', 1)
            if len(parts) > 1:
                text = parts[1]
        
        # Remove complete thinking blocks with tags
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<\|start_of_thought\|>.*?<\|end_of_thought\|>',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up orphaned opening tags
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'</?think>|</?thinking>|<\|start_of_thought\|>|<\|end_of_thought\|>', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _load_model(self):
        """Load OpenVINO GenAI pipeline (lazy loading)."""
        if self._model_loaded:
            return
        
        try:
            if self.verbose:
                print(f"[OpenVINOLLM] 🚀 Loading {self.model_path} on {self.device}...")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            xml_path = os.path.join(self.model_path, "openvino_model.xml")
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Model XML not found: {xml_path}")
            
            self.pipeline = ov_genai.LLMPipeline(self.model_path, self.device)
            self._model_loaded = True
            
            # Set generation config to prevent hangs
            config = self.pipeline.get_generation_config()
            config.max_new_tokens = 512  # Limit output length
            config.num_return_sequences = 1
            self.pipeline.set_generation_config(config)
            
            if self.verbose:
                print(f"[OpenVINOLLM] ✅ Model loaded successfully")
        
        except Exception as e:
            print(f"[OpenVINOLLM] ❌ Failed to load model: {e}")
            raise
    
    def warmup(self, prompt: str = "Hello, how are you?", max_tokens: int = 10):
        """Warm up the model with a dummy inference to compile/optimize the pipeline.
        
        This is useful on first run to:
        - Compile the model for the target device (NPU/GPU)
        - Initialize internal caches
        - Pre-allocate memory
        
        Args:
            prompt: Simple prompt for warmup (default: short greeting)
            max_tokens: Maximum tokens to generate (keep small for speed)
        """
        if not self._model_loaded:
            self._load_model()
        
        if self.verbose:
            print(f"[OpenVINOLLM] 🔥 Warming up model...")
        
        try:
            import openvino_genai as ov_genai
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            config.do_sample = False  # Greedy for speed
            
            # Run dummy inference (result is discarded)
            _ = self.pipeline.generate(prompt, config)
            
            if self.verbose:
                print(f"[OpenVINOLLM] ✅ Warmup complete")
        except Exception as e:
            if self.verbose:
                print(f"[OpenVINOLLM] ⚠️ Warmup failed (non-fatal): {e}")
    
    def invoke(
        self, 
        prompt: str, 
        timeout: int = 300,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512
    ) -> str:
        """Generate text from prompt (LangChain-compatible interface).
        
        Args:
            prompt: Input text prompt
            timeout: Generation timeout in seconds
            response_format: JSON schema for structured output (e.g., {"type": "object", ...})
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated text (or JSON string if response_format provided)
        """
        # Lazy load model on first call
        if not self._model_loaded:
            self._load_model()
        
        # Check response cache first
        cached_response = self.response_cache.get(prompt, system_prompt="")
        if cached_response is not None:
            if self.verbose:
                print(f"[OpenVINOLLM] ⚡ Cache HIT - returning cached response")
            return cached_response
        
        if self.verbose:
            print(f"[OpenVINOLLM] 🧠 Cache MISS - generating response...")
            import sys
            sys.stdout.flush()  # Force output to appear immediately
        
        try:
            # Set generation config
            import openvino_genai as ov_genai
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = max_new_tokens
            
            # Configure temperature and sampling
            if temperature < 0.01:
                config.do_sample = False  # Greedy decoding for deterministic output
            else:
                config.do_sample = True
                config.temperature = temperature
                config.top_p = 0.95
            
            final_prompt = prompt
            
            # Add JSON mode instruction if schema provided
            if response_format:
                json_instruction = (
                    "\n\nYou must respond with valid JSON matching this schema:\n"
                    f"{json.dumps(response_format, indent=2)}\n\n"
                    "Output only the JSON object, no additional text before or after."
                )
                final_prompt = final_prompt + json_instruction
            
            result = self.pipeline.generate(final_prompt, config)
            
            # Handle dict response format
            if isinstance(result, dict) and "text" in result:
                result = result["text"]
            
            # Strip thinking tags if requested
            if self.suppress_thinking:
                result = self._strip_thinking_tags(result)
            
            # Validate JSON if schema was provided
            if response_format:
                try:
                    # Extract JSON from potential markdown code blocks
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    elif "```" in result:
                        result = result.split("```")[1].split("```")[0].strip()
                    
                    # Validate it's valid JSON
                    json.loads(result)
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"[OpenVINOLLM] ⚠️ Invalid JSON generated, returning raw output: {e}")
            
            if self.verbose:
                print(f"[OpenVINOLLM] ✅ Generated {len(result)} characters")
            
            # Store in cache for future use
            self.response_cache.put(prompt, result, system_prompt="")
            
            return result
        
        except Exception as e:
            print(f"[OpenVINOLLM] ❌ Generation failed: {e}")
            raise


class RemoteLLM:
    """Remote LLM client - calls FastAPI server with same interface as OpenVINOLLM."""
    
    def __init__(self, server_url: str = "http://localhost:8000", verbose: bool = False, enable_cache: bool = False):
        """Initialize remote LLM client.
        
        Args:
            server_url: Base URL of the LLM server (e.g., "http://localhost:8000")
            verbose: Enable verbose logging
            enable_cache: Enable response caching
        """
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.response_cache = ResponseCache(enabled=enable_cache)  # Client-side cache
        self._check_server()
    
    def _check_server(self):
        """Check if server is reachable."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                if self.verbose:
                    print(f"[RemoteLLM] ✅ Connected to server: {self.server_url}")
                    print(f"[RemoteLLM] 📦 Server cache: {health.get('cache_enabled', False)}")
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to LLM server at {self.server_url}: {e}")
    
    def set_cache_prefix(self, prefix: str):
        """Set cache prefix (for compatibility)."""
        pass
    
    def start_fresh(self):
        """Reset cache on server."""
        try:
            requests.post(f"{self.server_url}/reset-cache", timeout=5)
            if self.verbose:
                print(f"[RemoteLLM] 🔄 Server cache reset")
        except Exception as e:
            if self.verbose:
                print(f"[RemoteLLM] ⚠️ Failed to reset cache: {e}")
    
    def invoke(
        self, 
        prompt: str, 
        timeout: int = 300,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512
    ) -> str:
        """Generate text from prompt by calling remote server.
        
        Args:
            prompt: Input text prompt
            timeout: Request timeout in seconds
            response_format: JSON schema for structured output
            temperature: Sampling temperature (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        # Check response cache first
        cached_response = self.response_cache.get(prompt, system_prompt="")
        if cached_response is not None:
            if self.verbose:
                print(f"[RemoteLLM] ⚡ Cache HIT - returning cached response")
            return cached_response
        
        if self.verbose:
            print(f"[RemoteLLM] 🌐 Cache MISS - sending request to server...")
        
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "response_format": response_format
            }
            
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            if self.verbose:
                print(f"[RemoteLLM] ✅ Response received ({len(result['response'])} chars)")
            
            # Store in cache
            self.response_cache.put(prompt, result["response"], system_prompt="")
            
            return result["response"]
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {timeout}s")
        except Exception as e:
            print(f"[RemoteLLM] ❌ Request failed: {e}")
            raise

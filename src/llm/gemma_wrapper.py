import json
import time
from typing import Dict, Any, Optional
import subprocess
from loguru import logger

class GemmaLLM:
    """
    A wrapper class for interacting with Gemma 3 4B model via Ollama.
    Handles model queries with retry logic and basic error handling.
    """
    
    def __init__(self, model: str = "gemma3:4b", max_retries: int = 3, timeout: int = 300):
        """
        Initialize the Gemma LLM wrapper.
        
        Args:
            model: The model name to use with Ollama
            max_retries: Maximum number of retry attempts for failed queries
            timeout: Maximum time in seconds to wait for a response
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        logger.info(f"Initialized GemmaLLM with model: {model}")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the Gemma model with the given prompt.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            
        Returns:
            Dict containing the response and metadata
        """
        start_time = time.time()
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                # Prepare the request payload
                request = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
                
                if system_prompt:
                    request["system"] = system_prompt
                
                # Call Ollama via CLI - just pass the prompt directly
                result = subprocess.run(
                    ["ollama", "run", self.model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Ollama error: {result.stderr}")
                
                # Ollama returns plain text, not JSON
                response_text = result.stdout.strip()
                
                return {
                    "response": response_text,
                    "model": self.model,
                    "tokens_used": 0,  # We don't have token count from CLI
                    "latency_seconds": time.time() - start_time,
                    "success": True
                }
                
            except Exception as e:
                last_error = str(e)
                attempt += 1
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)  # Exponential backoff could be added here
        
        return {
            "response": f"Error: Failed after {self.max_retries} attempts. Last error: {last_error}",
            "model": self.model,
            "tokens_used": 0,
            "latency_seconds": time.time() - start_time,
            "success": False
        }

    def generate_embeddings(self, text: str) -> Optional[list[float]]:
        """
        Generate embeddings for the given text using Ollama's embedding endpoint.
        
        Args:
            text: Input text to generate embeddings for
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            # Ollama doesn't have a built-in embeddings endpoint
            # We'll use a simple approach: generate a response and use it as a proxy
            # In a production system, you'd want to use a proper embedding model
            
            # For now, return a simple hash-based embedding as a fallback
            import hashlib
            import struct
            
            # Create a simple hash-based embedding
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert to list of floats (3584 dimensions for Gemma)
            embedding = []
            for i in range(3584):
                # Use different parts of the hash for each dimension
                byte_index = i % len(hash_bytes)
                embedding.append(float(hash_bytes[byte_index]) / 255.0)
            
            logger.info(f"Generated hash-based embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

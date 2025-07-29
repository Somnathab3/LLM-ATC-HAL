import ollama
import json
import time
import logging

class LLMClient:
    def __init__(self, model='llama3.1:8b', max_retries=3):
        self.client = ollama.Client()
        self.model = model
        self.max_retries = max_retries
        self.total_inference_time = 0.0
        self.inference_count = 0
        logging.basicConfig(level=logging.INFO)

    def ask(self, prompt, expect_json=False):
        """Ask the LLM a question with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat(
                    model=self.model, 
                    messages=[{'role': 'user', 'content': prompt}]
                )
                end_time = time.time()
                
                # Track inference timing
                inference_time = end_time - start_time
                self.total_inference_time += inference_time
                self.inference_count += 1
                
                content = response['message']['content'].strip()
                
                # If JSON is expected, try to parse it
                if expect_json:
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse JSON on attempt {attempt + 1}: {e}")
                        if attempt == self.max_retries - 1:
                            return {"error": "Invalid JSON response", "raw_content": content}
                        time.sleep(1)  # Brief delay before retry
                        continue
                
                return content
                
            except Exception as e:
                logging.error(f"LLM request failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return "Error: LLM unavailable"
                time.sleep(2)  # Longer delay for connection issues
        
        return "Error: Max retries exceeded"
    
    def get_average_inference_time(self):
        """Get average inference time per LLM call."""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count
    
    def get_total_inference_time(self):
        """Get total inference time across all calls."""
        return self.total_inference_time
    
    def get_inference_count(self):
        """Get total number of LLM calls made."""
        return self.inference_count
    
    def validate_response(self, response, expected_keys=None):
        """Validate LLM response format and content."""
        if isinstance(response, str) and response.startswith("Error:"):
            return False, response
        
        if expected_keys and isinstance(response, dict):
            missing_keys = [key for key in expected_keys if key not in response]
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"
        
        return True, "Valid response"
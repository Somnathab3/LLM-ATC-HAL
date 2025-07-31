import json
import logging
import time
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from functools import lru_cache

import ollama


@dataclass
class ChatMessage:
    """Represents a chat message with proper Ollama formatting"""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str


@dataclass
class LLMResponse:
    """Structured LLM response with performance metrics"""
    content: str
    response_time: float
    model: str
    cached: bool = False
    confidence: float = 0.0


class LLMClient:
    def __init__(self, model="llama3.1:8b", max_retries=2, timeout=15.0, enable_streaming=True, enable_caching=True, cache_size=128, enable_optimized_prompts=True) -> None:
        self.client = ollama.Client()
        self.model = model
        self.max_retries = max_retries  # Reduced from 3 for speed
        self.timeout = timeout  # Explicit timeout
        self.enable_streaming = enable_streaming
        self.enable_caching = enable_caching
        self.enable_optimized_prompts = enable_optimized_prompts
        self.total_inference_time = 0.0
        self.inference_count = 0
        self.cache_hits = 0
        self.function_call_enabled = True
        
        # Response cache
        if enable_caching:
            self._response_cache = {}
            self._cache_size = cache_size
        
        logging.basicConfig(level=logging.INFO)

    def create_chat_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[List[ChatMessage]] = None
    ) -> List[Dict[str, str]]:
        """
        Create properly formatted Ollama chat messages.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            context: Optional conversation context
            
        Returns:
            List of message dictionaries for Ollama
        """
        messages = []
        
        # System message (always first)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add context messages if provided
        if context:
            for msg in context:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # User message (current query)
        messages.append({
            "role": "user", 
            "content": user_prompt
        })
        
        return messages

    def ask(self, prompt, expect_json=False, enable_function_calls=True, system_prompt=None, priority="normal"):
        """Ask the LLM a question with retry logic and error handling - OPTIMIZED VERSION."""
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._create_cache_key(prompt, system_prompt)
            if cache_key in self._response_cache:
                self.cache_hits += 1
                cached_response = self._response_cache[cache_key]
                return cached_response
        
        # Adjust timeout based on priority
        timeout = self._get_priority_timeout(priority)
        
        for attempt in range(self.max_retries):
            try:
                # Enhanced prompt for function calling if enabled
                if enable_function_calls and self.function_call_enabled:
                    enhanced_prompt = self._enhance_prompt_for_function_calling(prompt)
                    user_prompt = enhanced_prompt
                else:
                    user_prompt = prompt

                # Create properly formatted messages with system/user separation
                default_system = system_prompt or "You are an expert Air Traffic Controller. Provide concise, accurate responses."
                messages = self.create_chat_messages(default_system, user_prompt)

                # Execute optimized request
                content = self._execute_chat_request(messages, timeout, expect_json)

                # Check for function calls in response
                if enable_function_calls and self.function_call_enabled:
                    function_call_result = self._process_function_calls(content)
                    if function_call_result:
                        # Cache successful response
                        if self.enable_caching and cache_key:
                            self._cache_response(cache_key, function_call_result)
                        return function_call_result

                # If JSON is expected, try to parse it
                if expect_json:
                    try:
                        parsed_content = json.loads(content)
                        # Cache successful response
                        if self.enable_caching and cache_key:
                            self._cache_response(cache_key, parsed_content)
                        return parsed_content
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse JSON on attempt {attempt + 1}: {e}")
                        if attempt == self.max_retries - 1:
                            error_response = {"error": "Invalid JSON response", "raw_content": content}
                            return error_response
                        time.sleep(0.5)  # Reduced delay
                        continue

                # Cache successful response
                if self.enable_caching and cache_key:
                    self._cache_response(cache_key, content)

                # Track performance
                response_time = time.time() - start_time
                self.total_inference_time += response_time
                self.inference_count += 1

                return content

            except Exception as e:
                logging.exception(f"LLM request failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return "Error: LLM unavailable"
                time.sleep(0.5)  # Reduced delay for connection issues

        return "Error: Max retries exceeded"

    def ask_optimized(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        expect_json: bool = False,
        context: Optional[List[ChatMessage]] = None,
        priority: str = "normal"
    ) -> LLMResponse:
        """
        High-performance LLM query with proper chat formatting.
        
        Args:
            user_prompt: User question/request
            system_prompt: System instructions (ATC role, requirements)
            expect_json: Whether to expect JSON response
            context: Conversation context
            priority: Request priority for timeout adjustment
            
        Returns:
            LLMResponse with content and performance metrics
        """
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._create_cache_key(user_prompt, system_prompt)
            if cache_key in self._response_cache:
                self.cache_hits += 1
                cached_response = self._response_cache[cache_key]
                return LLMResponse(
                    content=cached_response,
                    response_time=time.time() - start_time,
                    model=self.model,
                    cached=True
                )
        
        # Adjust timeout based on priority
        timeout = self._get_priority_timeout(priority)
        
        # Create properly formatted messages
        default_system = system_prompt or "You are an expert Air Traffic Controller. Provide concise, accurate responses."
        messages = self.create_chat_messages(default_system, user_prompt, context)
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                response = self._execute_chat_request(messages, timeout, expect_json)
                
                # Cache successful response
                if self.enable_caching and cache_key:
                    self._cache_response(cache_key, response)
                
                # Track performance
                response_time = time.time() - start_time
                self.total_inference_time += response_time
                self.inference_count += 1
                
                return LLMResponse(
                    content=response,
                    response_time=response_time,
                    model=self.model,
                    cached=False
                )
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return LLMResponse(
                        content=f"Error: {str(e)}",
                        response_time=time.time() - start_time,
                        model=self.model
                    )
                time.sleep(0.5)  # Brief delay, reduced from 1-2s
        
        return LLMResponse(
            content="Error: Max retries exceeded",
            response_time=time.time() - start_time,
            model=self.model
        )

    def _execute_chat_request(
        self,
        messages: List[Dict[str, str]],
        timeout: float,
        expect_json: bool
    ) -> str:
        """Execute the actual chat request to Ollama"""
        
        options = {
            "temperature": 0.1,  # Low for consistency
            "top_p": 0.9,
            "num_predict": 300,  # Reduced token limit for speed
        }
        
        if self.enable_streaming:
            # Use streaming for faster response times
            response_chunks = []
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if chunk['message']['content']:
                    response_chunks.append(chunk['message']['content'])
            
            content = ''.join(response_chunks).strip()
        else:
            # Standard non-streaming request - UPDATED TO USE PROPER MESSAGES
            response = self.client.chat(
                model=self.model,
                messages=messages,  # Now properly formatted with system/user roles
                options=options
            )
            content = response["message"]["content"].strip()
        
        # Handle JSON parsing
        if expect_json:
            return self._parse_json_response_fast(content)
        
        return content

    def _enhance_prompt_for_function_calling(self, original_prompt: str) -> str:
        """Enhance prompt to include function calling instructions"""
        function_instructions = """
Available Functions:
- GetAllAircraftInfo(): Get information about all aircraft in simulation
- GetConflictInfo(): Get information about current conflicts
- ContinueMonitoring(): Continue monitoring without action
- SendCommand(command): Send a BlueSky command (e.g., "ALT AAL123 FL350")
- SearchExperienceLibrary(scenario_type, similarity_threshold): Search for similar scenarios
- GetWeatherInfo(lat, lon): Get weather information
- GetAirspaceInfo(): Get airspace restrictions

To call a function, respond with JSON format:
{
    "function_call": {
        "name": "FunctionName",
        "arguments": {"param1": "value1", "param2": "value2"}
    }
}

Or provide a regular text response for analysis and reasoning.

"""
        return function_instructions + "\n" + original_prompt

    def _process_function_calls(self, content: str) -> Optional[dict[str, Any]]:
        """Process function calls from LLM response"""
        try:
            # Try to parse as JSON to detect function calls
            parsed_content = json.loads(content)

            if isinstance(parsed_content, dict) and "function_call" in parsed_content:
                function_call = parsed_content["function_call"]
                function_name = function_call.get("name")
                arguments = function_call.get("arguments", {})

                logging.info(f"Function call detected: {function_name} with args {arguments}")

                # Execute the function call
                result = self._execute_function_call(function_name, arguments)

                # Return structured response
                return {
                    "type": "function_call",
                    "function_name": function_name,
                    "arguments": arguments,
                    "result": result,
                    "original_content": content,
                }

        except json.JSONDecodeError:
            # Not a JSON response, continue with normal processing
            pass
        except Exception as e:
            logging.exception(f"Error processing function call: {e}")

        return None

    def _execute_function_call(
        self, function_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a function call and return the result"""
        try:
            # TODO: Function calling disabled to avoid circular imports
            # Need to refactor if function calling is required
            return {
                "success": False,
                "error": f"Function calling disabled: {function_name}",
                "timestamp": time.time(),
            }

        except Exception as e:
            logging.exception(f"Error executing function {function_name}: {e}")
            return {
                "success": False,
                "function_name": function_name,
                "error": str(e),
                "timestamp": time.time(),
            }

    def chat_with_function_calling(
        self, messages: list[dict[str, str]], max_function_calls: int = 5,
    ) -> dict[str, Any]:
        """
        Extended chat interface with function calling support

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_function_calls: Maximum number of function calls allowed in a single chat

        Returns:
            Final response with function call history
        """
        conversation_history = messages.copy()
        function_calls_made = []

        for _call_count in range(max_function_calls):
            # Create prompt from conversation history
            prompt = self._format_conversation_for_prompt(conversation_history)

            # Get LLM response
            response = self.ask(prompt, enable_function_calls=True)

            # Check if it's a function call
            if isinstance(response, dict) and response.get("type") == "function_call":
                function_calls_made.append(response)

                # Add function call and result to conversation
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": f"Called function {response['function_name']} with args {response['arguments']}",
                    },
                )
                conversation_history.append(
                    {
                        "role": "function",
                        "content": json.dumps(response["result"], default=str),
                    },
                )

                # Continue the conversation
                continue
            # Regular response, end the function calling loop
            return {
                "final_response": response,
                "function_calls": function_calls_made,
                "conversation_history": conversation_history,
                "total_function_calls": len(function_calls_made),
            }

        # Max function calls reached
        return {
            "final_response": "Maximum function calls reached",
            "function_calls": function_calls_made,
            "conversation_history": conversation_history,
            "total_function_calls": len(function_calls_made),
            "error": "max_function_calls_exceeded",
        }

    def _format_conversation_for_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format conversation history into a single prompt"""
        formatted_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            elif role == "function":
                formatted_parts.append(f"Function Result: {content}")

        return "\n".join(formatted_parts)

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

    # === OPTIMIZATION METHODS ===

    def _parse_json_response_fast(self, content: str) -> str:
        """Fast JSON parsing with minimal fallback"""
        try:
            # Quick validation and return
            json.loads(content)
            return content
        except json.JSONDecodeError:
            # Simple JSON extraction - no complex cleaning
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            if json_match:
                try:
                    json.loads(json_match.group(0))
                    return json_match.group(0)
                except:
                    pass
            
            # Return error structure quickly
            return json.dumps({
                "error": "Invalid JSON response",
                "raw_content": content[:200]  # Truncated
            })

    def _create_cache_key(self, user_prompt: str, system_prompt: Optional[str]) -> str:
        """Create cache key from prompts"""
        key_parts = [user_prompt]
        if system_prompt:
            key_parts.append(system_prompt)
        return hash('|'.join(key_parts).__str__())

    def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache response with size limit"""
        if not hasattr(self, '_response_cache'):
            return
            
        if len(self._response_cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = response

    def _get_priority_timeout(self, priority: str) -> float:
        """Get timeout based on request priority"""
        timeouts = {
            'low': self.timeout * 0.5,
            'normal': self.timeout,
            'high': self.timeout * 1.5
        }
        return timeouts.get(priority, self.timeout)

    # Optimized prompt templates for ATC
    def get_conflict_resolution_system_prompt(self) -> str:
        """Concise system prompt for conflict resolution"""
        return """You are an expert Air Traffic Controller. Your task: resolve aircraft conflicts safely and efficiently.

CRITICAL REQUIREMENTS:
- Maintain 5 NM horizontal OR 1000 ft vertical separation
- Minimize flight path disruption
- ICAO compliance mandatory

RESPONSE FORMAT REQUIRED:
COMMAND: [HDG/ALT/SPD] [AIRCRAFT_ID] [VALUE]
RATIONALE: [Brief explanation]
CONFIDENCE: [0.0-1.0]

Examples:
COMMAND: HDG UAL890 045
RATIONALE: Right turn avoids conflict
CONFIDENCE: 0.92"""

    def get_conflict_detection_system_prompt(self) -> str:
        """Concise system prompt for conflict detection"""
        return """You are a precision conflict detection specialist.

DETECTION RULES:
- Conflict = BOTH conditions violated: horizontal <5 NM AND vertical <1000 ft
- Calculate actual distances, don't estimate
- Only detect with mathematical certainty

RESPONSE: JSON format
{
  "conflict_detected": true/false,
  "aircraft_pairs": ["AC1-AC2"],
  "confidence": 0.0-1.0,
  "analysis": "Brief calculation summary"
}"""

    # Performance metrics
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        avg_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 else 0.0
        )
        
        return {
            "average_response_time": avg_time,
            "total_requests": self.inference_count,
            "cache_hit_rate": (
                self.cache_hits / self.inference_count 
                if self.inference_count > 0 else 0.0
            ),
            "total_inference_time": self.total_inference_time,
            "model": self.model
        }

    def reset_stats(self) -> None:
        """Reset performance tracking"""
        self.total_inference_time = 0.0
        self.inference_count = 0
        self.cache_hits = 0


# Utility functions for quick ATC queries
def quick_conflict_resolution(
    aircraft_1: Dict[str, Any],
    aircraft_2: Dict[str, Any],
    time_to_conflict: float,
    client: Optional[LLMClient] = None
) -> LLMResponse:
    """
    Quick conflict resolution with minimal prompt overhead.
    
    Args:
        aircraft_1: First aircraft data
        aircraft_2: Second aircraft data  
        time_to_conflict: Time to conflict in seconds
        client: Optional client instance
        
    Returns:
        LLMResponse with resolution command
    """
    if not client:
        client = LLMClient()
    
    # Minimal user prompt
    user_prompt = f"""CONFLICT: {aircraft_1.get('id', 'AC1')} and {aircraft_2.get('id', 'AC2')}
Time to conflict: {time_to_conflict:.1f}s

AC1: {aircraft_1.get('lat', 0):.3f}°N, {aircraft_1.get('lon', 0):.3f}°E, {aircraft_1.get('alt', 0):.0f}ft, {aircraft_1.get('hdg', 0):.0f}°
AC2: {aircraft_2.get('lat', 0):.3f}°N, {aircraft_2.get('lon', 0):.3f}°E, {aircraft_2.get('alt', 0):.0f}ft, {aircraft_2.get('hdg', 0):.0f}°

Provide resolution command:"""
    
    return client.ask_optimized(
        user_prompt=user_prompt,
        system_prompt=client.get_conflict_resolution_system_prompt(),
        priority="high"
    )


def quick_conflict_detection(
    aircraft_states: List[Dict[str, Any]],
    client: Optional[LLMClient] = None
) -> LLMResponse:
    """
    Quick conflict detection with minimal overhead.
    
    Args:
        aircraft_states: List of aircraft data
        client: Optional client instance
        
    Returns:
        LLMResponse with detection results (JSON)
    """
    if not client:
        client = LLMClient()
    
    # Compact aircraft representation
    aircraft_summary = []
    for i, ac in enumerate(aircraft_states):
        summary = f"AC{i+1}: {ac.get('lat', 0):.3f},{ac.get('lon', 0):.3f},{ac.get('alt', 0):.0f}ft,{ac.get('hdg', 0):.0f}°"
        aircraft_summary.append(summary)
    
    user_prompt = f"""Aircraft positions:
{chr(10).join(aircraft_summary)}

Detect conflicts (5NM/1000ft rule):"""
    
    return client.ask_optimized(
        user_prompt=user_prompt,
        system_prompt=client.get_conflict_detection_system_prompt(),
        expect_json=True,
        priority="high"
    )

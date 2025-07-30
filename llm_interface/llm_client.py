import json
import logging
import time
from typing import Any, Optional

import ollama

# Import tools for function calling
from llm_atc.tools import bluesky_tools


class LLMClient:
    def __init__(self, model="llama3.1:8b", max_retries=3) -> None:
        self.client = ollama.Client()
        self.model = model
        self.max_retries = max_retries
        self.total_inference_time = 0.0
        self.inference_count = 0
        self.function_call_enabled = True
        logging.basicConfig(level=logging.INFO)

    def ask(self, prompt, expect_json=False, enable_function_calls=True):
        """Ask the LLM a question with retry logic and error handling."""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Enhanced prompt for function calling if enabled
                if enable_function_calls and self.function_call_enabled:
                    enhanced_prompt = self._enhance_prompt_for_function_calling(prompt)
                else:
                    enhanced_prompt = prompt

                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                )
                end_time = time.time()

                # Track inference timing
                inference_time = end_time - start_time
                self.total_inference_time += inference_time
                self.inference_count += 1

                content = response["message"]["content"].strip()

                # Check for function calls in response
                if enable_function_calls and self.function_call_enabled:
                    function_call_result = self._process_function_calls(content)
                    if function_call_result:
                        return function_call_result

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
                logging.exception(f"LLM request failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return "Error: LLM unavailable"
                time.sleep(2)  # Longer delay for connection issues

        return "Error: Max retries exceeded"

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

    def _execute_function_call(self, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a function call and return the result"""
        try:
            # Dispatch to the appropriate tool function
            if function_name in bluesky_tools.TOOL_REGISTRY:
                tool_function = bluesky_tools.TOOL_REGISTRY[function_name]
                result = tool_function(**arguments)

                return {
                    "success": True,
                    "function_name": function_name,
                    "result": result,
                    "timestamp": time.time(),
                }
            return {
                "success": False,
                "error": f"Unknown function: {function_name}",
                "available_functions": list(bluesky_tools.TOOL_REGISTRY.keys()),
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

    def chat_with_function_calling(self, messages: list[dict[str, str]],
                                 max_function_calls: int = 5) -> dict[str, Any]:
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
                conversation_history.append({
                    "role": "assistant",
                    "content": f"Called function {response['function_name']} with args {response['arguments']}",
                })
                conversation_history.append({
                    "role": "function",
                    "content": json.dumps(response["result"], default=str),
                })

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

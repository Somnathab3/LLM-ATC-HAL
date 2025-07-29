# testing/result_streamer.py
"""
Result Streaming Module for LLM-ATC-HAL Framework
Handles memory-efficient streaming of test results to disk
"""

import json
import threading
import queue
import logging
import os
from typing import Dict, Any, Optional
from dataclasses import asdict
from contextlib import contextmanager

from .test_executor import TestResult


class ResultStreamer:
    """Memory-efficient result streaming to handle large test batches"""
    
    def __init__(self, output_file: str, buffer_size: int = 100):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.result_queue = queue.Queue(maxsize=buffer_size * 2)
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.results_written = 0
        self.errors_encountered = 0
    
    def start(self):
        """Start the background result streaming thread"""
        if self.writer_thread and self.writer_thread.is_alive():
            self.logger.warning("Result streamer already running")
            return
        
        self.stop_event.clear()
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
        self.logger.info(f"Started result streaming to {self.output_file}")
    
    def stop(self, timeout: float = 10.0):
        """Stop the result streaming thread"""
        if not self.writer_thread or not self.writer_thread.is_alive():
            return
        
        self.stop_event.set()
        self.writer_thread.join(timeout=timeout)
        
        if self.writer_thread.is_alive():
            self.logger.warning("Result streamer did not stop gracefully")
        else:
            self.logger.info(f"Result streamer stopped. Written: {self.results_written}, Errors: {self.errors_encountered}")
    
    def stream_result(self, result: TestResult):
        """Stream a test result (non-blocking)"""
        try:
            # Convert result to dict for JSON serialization
            result_dict = asdict(result)
            self.result_queue.put(result_dict, block=False)
        except queue.Full:
            self.logger.warning("Result queue full, dropping result")
            self.errors_encountered += 1
        except Exception as e:
            self.logger.error(f"Error queuing result: {e}")
            self.errors_encountered += 1
    
    def _writer_worker(self):
        """Background worker thread that writes results to disk"""
        buffer = []
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        try:
            with open(self.output_file, 'w') as f:
                # Write initial metadata
                metadata = {
                    "format": "jsonl",
                    "content": "llm_atc_hal_test_results",
                    "version": "1.0"
                }
                f.write(json.dumps(metadata) + '\n')
                f.flush()
                
                while not self.stop_event.is_set() or not self.result_queue.empty():
                    try:
                        # Get result with timeout to allow checking stop event
                        result = self.result_queue.get(timeout=1.0)
                        buffer.append(result)
                        
                        # Write buffer when full or on stop
                        if len(buffer) >= self.buffer_size or self.stop_event.is_set():
                            self._flush_buffer(f, buffer)
                            buffer.clear()
                        
                        self.result_queue.task_done()
                        
                    except queue.Empty:
                        # Timeout - continue to check stop event
                        if buffer and self.stop_event.is_set():
                            self._flush_buffer(f, buffer)
                            buffer.clear()
                        continue
                    except (OSError, IOError) as e:
                        self.logger.error(f"IO error writing results: {e}")
                        self.errors_encountered += 1
                        break
                    except Exception as e:
                        self.logger.exception(f"Unexpected error in result writer: {e}")
                        self.errors_encountered += 1
                        continue
                
                # Flush any remaining results
                if buffer:
                    self._flush_buffer(f, buffer)
        
        except Exception as e:
            self.logger.exception(f"Fatal error in result writer: {e}")
            self.errors_encountered += 1
    
    def _flush_buffer(self, file_handle, buffer):
        """Flush buffer to disk"""
        try:
            for result in buffer:
                file_handle.write(json.dumps(result, default=str) + '\n')
            file_handle.flush()
            self.results_written += len(buffer)
            
            if self.results_written % 1000 == 0:
                self.logger.info(f"Streamed {self.results_written} results")
                
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {e}")
            self.errors_encountered += 1
    
    @contextmanager
    def streaming_context(self):
        """Context manager for result streaming"""
        try:
            self.start()
            yield self
        finally:
            self.stop()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get streaming statistics"""
        return {
            'results_written': self.results_written,
            'errors_encountered': self.errors_encountered,
            'queue_size': self.result_queue.qsize() if hasattr(self.result_queue, 'qsize') else 0
        }


class BatchResultProcessor:
    """Process results in batches to manage memory usage"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def process_results_in_batches(self, results_file: str, processor_func, *args, **kwargs):
        """Process results file in batches to manage memory"""
        try:
            batch = []
            line_count = 0
            
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        # Skip metadata line
                        if line_count == 0:
                            line_count += 1
                            continue
                        
                        result_data = json.loads(line.strip())
                        batch.append(result_data)
                        
                        if len(batch) >= self.batch_size:
                            processor_func(batch, *args, **kwargs)
                            batch.clear()
                        
                        line_count += 1
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_count}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing line {line_count}: {e}")
                        continue
                
                # Process remaining results
                if batch:
                    processor_func(batch, *args, **kwargs)
                
                self.logger.info(f"Processed {line_count - 1} results in batches of {self.batch_size}")
                
        except FileNotFoundError:
            self.logger.error(f"Results file not found: {results_file}")
        except Exception as e:
            self.logger.exception(f"Error processing results file: {e}")
    
    def aggregate_batch_statistics(self, batch: list) -> Dict[str, Any]:
        """Aggregate statistics for a batch of results"""
        if not batch:
            return {}
        
        # Convert to simple statistics
        total_tests = len(batch)
        successful_tests = sum(1 for result in batch if not result.get('errors', []))
        
        response_times = [result.get('response_time', 0) for result in batch if result.get('response_time')]
        safety_margins = [result.get('safety_margin', 0) for result in batch if result.get('safety_margin')]
        hallucinations = sum(1 for result in batch if result.get('hallucination_detected', False))
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'error_rate': (total_tests - successful_tests) / total_tests if total_tests > 0 else 0,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'avg_safety_margin': sum(safety_margins) / len(safety_margins) if safety_margins else 0,
            'hallucination_rate': hallucinations / total_tests if total_tests > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    import time
    
    # Test result streaming
    output_file = "test_results_stream.jsonl"
    
    with ResultStreamer(output_file).streaming_context() as streamer:
        # Simulate streaming results
        for i in range(10):
            test_result = TestResult(
                test_id=f"test_{i}",
                scenario_type="test",
                complexity_level="simple",
                model_used="test_model",
                response_time=0.5,
                processing_time=1.0,
                hallucination_detected=False,
                hallucination_types=[],
                confidence_score=0.8,
                safety_margin=0.7,
                icao_compliant=True,
                accuracy=0.9,
                precision=0.85,
                recall=0.88,
                scenario_data={},
                llm_response={},
                ground_truth={},
                errors=[]
            )
            streamer.stream_result(test_result)
            time.sleep(0.1)
    
    print(f"Streaming complete. Check {output_file}")

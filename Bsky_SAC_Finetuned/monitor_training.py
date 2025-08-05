#!/usr/bin/env python3
"""
Training Monitor - Track GPU usage and training progress
"""

import time
import subprocess
import json
import os
from pathlib import Path

def get_gpu_stats():
    """Get GPU memory and utilization stats."""
    try:
        # Run nvidia-smi to get GPU stats
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            stats = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    stats.append({
                        'gpu_util': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3])
                    })
            return stats
        else:
            return None
    except Exception:
        return None

def get_training_stats(log_file):
    """Parse training log for current stats."""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for recent progress information
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "[PROGRESS]" in line and "Step" in line:
                # Extract step and loss info
                parts = line.split()
                for i, part in enumerate(parts):
                    if "Step" in part and i+1 < len(parts):
                        try:
                            step = int(parts[i+1].rstrip(':'))
                            # Look for loss in the same line
                            for j in range(i, len(parts)):
                                if "Loss:" in parts[j] and j+1 < len(parts):
                                    loss = float(parts[j+1])
                                    return {"step": step, "loss": loss}
                        except (ValueError, IndexError):
                            continue
        return None
    except Exception:
        return None

def monitor_training(log_dir="models/test_run/logs", interval=30):
    """Monitor training progress and GPU usage."""
    print("🔍 Training Monitor Started")
    print("=" * 60)
    print(f"Monitoring log directory: {log_dir}")
    print(f"Update interval: {interval} seconds")
    print("=" * 60)
    
    log_file = Path(log_dir) / "sac_training.log"
    start_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Get GPU stats
        gpu_stats = get_gpu_stats()
        
        # Get training stats
        training_stats = get_training_stats(log_file)
        
        # Display stats
        print(f"\n⏰ Elapsed: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s")
        
        if gpu_stats:
            for i, gpu in enumerate(gpu_stats):
                memory_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"🎮 GPU {i}: {gpu['gpu_util']:3d}% util | "
                      f"{gpu['memory_used']:5d}/{gpu['memory_total']:5d}MB ({memory_pct:4.1f}%) | "
                      f"{gpu['temperature']:2d}°C")
        else:
            print("🎮 GPU: Stats unavailable")
        
        if training_stats:
            print(f"🚂 Training: Step {training_stats['step']:,} | Loss {training_stats['loss']:.6f}")
        else:
            print("🚂 Training: Waiting for progress...")
        
        print("-" * 60)
        
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
            break

def main():
    """Main monitor function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training progress and GPU usage")
    parser.add_argument("--log-dir", default="models/test_run/logs", 
                       help="Training log directory to monitor")
    parser.add_argument("--interval", type=int, default=30,
                       help="Update interval in seconds")
    
    args = parser.parse_args()
    
    try:
        monitor_training(args.log_dir, args.interval)
    except Exception as e:
        print(f"❌ Monitor error: {e}")

if __name__ == "__main__":
    main()

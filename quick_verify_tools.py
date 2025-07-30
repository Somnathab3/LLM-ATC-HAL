#!/usr/bin/env python3
"""
Quick verification that extended BlueSky tools are working
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from llm_atc.tools import (
        get_distance, 
        step_simulation, 
        reset_simulation, 
        get_minimum_separation, 
        check_separation_violation,
        TOOL_REGISTRY
    )
    
    print("✅ All new tools imported successfully!")
    print(f"📊 Total tools in registry: {len(TOOL_REGISTRY)}")
    
    # Quick test
    min_sep = get_minimum_separation()
    print(f"📐 Standard separation: {min_sep['horizontal_nm']} nm / {min_sep['vertical_ft']} ft")
    
    print("🎉 Extended BlueSky tools are ready for Monte Carlo runner!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

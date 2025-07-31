#!/usr/bin/env python3
"""
Dependency Analysis Tool for LLM-ATC-HAL
========================================

Analyzes the codebase to identify:
1. Core files that are actively used by the system
2. Orphaned files that may be safe to remove
3. Import relationships and dependency chains

This helps clean up the codebase by identifying unused legacy files.
"""

import os
import ast
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Set, Dict, Any


def extract_imports(file_path: str) -> List[str]:
    """Extract all imports from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for alias in node.names:
                    if module:
                        imports.append(f'{module}.{alias.name}')
                    else:
                        imports.append(alias.name)
        
        return imports
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []


def find_all_python_files() -> List[str]:
    """Find all Python files in the project, excluding certain directories"""
    all_python_files = []
    skip_patterns = [
        'venv', '__pycache__', '.git', 
        'experiments/results', 'experiments/monte_carlo', 
        'experiments/quick_test', 'experiments/detection', 
        'experiments/comprehensive', 'experiments/clean_benchmark',
        '.egg-info'
    ]
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in skip_patterns):
            continue
            
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file))
                # Normalize path separators to forward slashes
                normalized_path = rel_path.replace('\\', '/')
                all_python_files.append(normalized_path)
    
    return all_python_files


def build_import_graph(all_files: List[str]) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build import dependency graph"""
    import_graph = defaultdict(set)  # file -> files it imports
    reverse_import_graph = defaultdict(set)  # file -> files that import it
    
    for file_path in all_files:
        imports = extract_imports(file_path)
        
        for imp in imports:
            # Only track project-internal imports
            if imp.startswith(('llm_atc', 'scenarios', 'llm_interface', 'analysis', 'solver', 'bluesky_sim')):
                # Map import to potential file paths
                imp_parts = imp.split('.')
                potential_files = []
                
                # Try different combinations to find the actual file
                for i in range(1, len(imp_parts) + 1):
                    # Try as direct .py file
                    potential_path = '/'.join(imp_parts[:i]) + '.py'
                    if potential_path in all_files:
                        potential_files.append(potential_path)
                    
                    # Try as package with __init__.py
                    potential_init = '/'.join(imp_parts[:i]) + '/__init__.py'
                    if potential_init in all_files:
                        potential_files.append(potential_init)
                
                # Add the import relationships
                for potential_file in potential_files:
                    import_graph[file_path].add(potential_file)
                    reverse_import_graph[potential_file].add(file_path)
    
    return import_graph, reverse_import_graph


def find_reachable_files(import_graph: Dict[str, Set[str]], entry_points: List[str], known_core: Set[str]) -> Set[str]:
    """Find all files reachable from entry points using BFS"""
    reachable = set()
    queue = deque()
    
    # Start from entry points
    for entry_point in entry_points:
        if entry_point in import_graph or any(entry_point in files for files in import_graph.values()):
            reachable.add(entry_point)
            queue.append(entry_point)
    
    # Add known core files
    for core_file in known_core:
        if core_file not in reachable:
            reachable.add(core_file)
            queue.append(core_file)
    
    # BFS to find all reachable files
    while queue:
        current = queue.popleft()
        for imported_file in import_graph.get(current, []):
            if imported_file not in reachable:
                reachable.add(imported_file)
                queue.append(imported_file)
    
    return reachable


def analyze_orphaned_files(all_files: List[str], reachable: Set[str], reverse_graph: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    """Analyze files that appear to be orphaned"""
    orphaned = []
    
    for file_path in all_files:
        if file_path not in reachable:
            # Categorize the file
            is_test = 'test' in file_path.lower()
            is_example = any(keyword in file_path.lower() for keyword in ['example', 'demo', 'manual'])
            is_standalone = any(keyword in file_path.lower() for keyword in ['enhanced_', 'system_', 'run_'])
            is_config = file_path.endswith(('.yaml', '.yml', '.json', '.toml'))
            
            # Check if it's imported by any files (even orphaned ones)
            imported_by = list(reverse_graph.get(file_path, []))
            
            # Try to determine purpose from filename/path
            purpose = "Unknown"
            if is_test:
                purpose = "Test file"
            elif is_example:
                purpose = "Example/Demo script"
            elif is_standalone:
                purpose = "Standalone utility"
            elif 'baseline' in file_path:
                purpose = "Baseline model"
            elif 'memory' in file_path:
                purpose = "Memory system component"
            elif 'experiment' in file_path:
                purpose = "Experimental feature"
            
            orphaned.append({
                'file': file_path,
                'purpose': purpose,
                'is_test': is_test,
                'is_example': is_example,
                'is_standalone': is_standalone,
                'is_config': is_config,
                'imported_by': imported_by,
                'import_count': len(imported_by)
            })
    
    return orphaned


def main():
    """Main analysis function"""
    print("🔍 LLM-ATC-HAL Dependency Analysis")
    print("=" * 50)
    
    # Define entry points and known core files
    entry_points = ['cli.py']
    
    # Files we know are core to the system
    known_core_files = {
        'cli.py',
        'scenarios/monte_carlo_runner.py',
        'scenarios/scenario_generator.py',
        'scenarios/monte_carlo_framework.py',
        'llm_atc/tools/llm_prompt_engine.py',
        'llm_atc/tools/bluesky_tools.py',
        'llm_interface/llm_client.py',
        'analysis/enhanced_hallucination_detection.py',
        'analysis/metrics.py',
        'llm_atc/experiments/distribution_shift_runner.py'
    }
    
    # Step 1: Find all Python files
    print("📁 Finding all Python files...")
    all_python_files = find_all_python_files()
    print(f"   Found {len(all_python_files)} Python files")
    
    # Step 2: Build import graph
    print("🔗 Building import dependency graph...")
    import_graph, reverse_import_graph = build_import_graph(all_python_files)
    print(f"   Mapped {len(import_graph)} files with dependencies")
    
    # Step 3: Find reachable files
    print("🎯 Finding core files reachable from entry points...")
    reachable = find_reachable_files(import_graph, entry_points, known_core_files)
    print(f"   Found {len(reachable)} core files")
    
    # Step 4: Analyze orphaned files
    print("🔍 Analyzing potentially orphaned files...")
    orphaned = analyze_orphaned_files(all_python_files, reachable, reverse_import_graph)
    print(f"   Found {len(orphaned)} potentially orphaned files")
    
    # Print results
    print("\n" + "=" * 60)
    print("📋 ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\n✅ CORE FILES ({len(reachable)} files):")
    print("   These files are actively used by the system")
    for file_path in sorted(reachable):
        print(f"   ✅ {file_path}")
    
    print(f"\n❓ POTENTIALLY ORPHANED FILES ({len(orphaned)} files):")
    print("   These files may be safe to remove")
    
    # Group orphaned files by category
    categories = {}
    for item in orphaned:
        purpose = item['purpose']
        if purpose not in categories:
            categories[purpose] = []
        categories[purpose].append(item)
    
    for purpose, files in categories.items():
        print(f"\n   📂 {purpose} ({len(files)} files):")
        for item in sorted(files, key=lambda x: x['file']):
            file_path = item['file']
            tags = []
            
            if item['is_test']:
                tags.append('TEST')
            if item['is_example']:
                tags.append('EXAMPLE')
            if item['is_standalone']:
                tags.append('STANDALONE')
            if item['imported_by']:
                tags.append(f'IMPORTED_BY:{item["import_count"]}')
            
            tag_str = f' [{", ".join(tags)}]' if tags else ''
            print(f"      ❓ {file_path}{tag_str}")
            
            # Show what imports this file (if any)
            if item['imported_by']:
                for importer in sorted(item['imported_by'])[:3]:  # Show max 3
                    print(f"         ← imported by: {importer}")
                if len(item['imported_by']) > 3:
                    print(f"         ← ... and {len(item['imported_by']) - 3} more")
    
    # Summary and recommendations
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print(f"✅ Core files (keep):     {len(reachable)}")
    print(f"❓ Potentially orphaned:  {len(orphaned)}")
    print(f"📈 Total Python files:   {len(all_python_files)}")
    
    # Specific recommendations
    safe_to_remove = [item for item in orphaned if not item['imported_by'] and (item['is_example'] or item['is_standalone'])]
    needs_review = [item for item in orphaned if item['imported_by'] or not (item['is_example'] or item['is_standalone'])]
    
    print(f"\n🔒 SAFE TO REMOVE ({len(safe_to_remove)} files):")
    print("   These files are not imported by anything and appear to be examples/utilities")
    for item in safe_to_remove:
        print(f"   🗑️  {item['file']} - {item['purpose']}")
    
    print(f"\n⚠️  NEEDS REVIEW ({len(needs_review)} files):")
    print("   These files may have dependencies or unclear usage")
    for item in needs_review:
        print(f"   ⚠️  {item['file']} - {item['purpose']}")
        if item['imported_by']:
            print(f"       Imported by {len(item['imported_by'])} files")


if __name__ == "__main__":
    main()

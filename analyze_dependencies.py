#!/usr/bin/env python3
"""
Dependency Analysis Script
Analyzes all Python files to build a dependency graph
"""
import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict

class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.from_imports = []
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.from_imports.append(node.module)

def analyze_file(file_path: Path) -> tuple[list, list]:
    """Analyze a Python file and return its imports"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        return analyzer.imports, analyzer.from_imports
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return [], []

def is_local_import(import_name: str, project_modules: Set[str]) -> bool:
    """Check if an import is a local project module"""
    # Handle relative imports
    if import_name.startswith('.'):
        return True
    
    # Check if it starts with any project module
    for module in project_modules:
        if import_name.startswith(module):
            return True
    
    return False

def main():
    project_root = Path('.')
    
    # Find all Python files, excluding virtual environment and other unwanted directories
    exclude_dirs = {'venv', '.git', '__pycache__', '.pytest_cache', 'node_modules', '.vscode'}
    python_files = []
    
    for py_file in project_root.rglob('*.py'):
        # Skip if in excluded directory
        if any(part in exclude_dirs for part in py_file.parts):
            continue
        python_files.append(py_file)
    
    # Identify project modules (top-level directories with __init__.py)
    project_modules = set()
    for path in project_root.iterdir():
        if path.is_dir() and path.name not in exclude_dirs and (path / '__init__.py').exists():
            project_modules.add(path.name)
    
    # Also add root-level modules
    for py_file in project_root.glob('*.py'):
        if py_file.name != '__init__.py':
            project_modules.add(py_file.stem)
    
    print(f"Found {len(python_files)} Python files")
    print(f"Project modules: {sorted(project_modules)}")
    
    # Analyze dependencies
    file_deps = {}
    all_imports = defaultdict(list)
    
    for py_file in python_files:
        rel_path = py_file.relative_to(project_root)
        imports, from_imports = analyze_file(py_file)
        
        # Combine all imports
        all_file_imports = imports + from_imports
        local_imports = [imp for imp in all_file_imports if is_local_import(imp, project_modules)]
        
        file_deps[str(rel_path)] = local_imports
        
        for imp in local_imports:
            all_imports[imp].append(str(rel_path))
    
    # Print dependency analysis
    print("\n=== DEPENDENCY ANALYSIS ===")
    
    # CLI dependencies
    print("\n1. CLI Dependencies (cli.py):")
    cli_deps = file_deps.get('cli.py', [])
    print(f"  Direct imports: {cli_deps}")
    
    # BSKY_GYM_LLM dependencies  
    print("\n2. BSKY_GYM_LLM Dependencies:")
    bsky_files = [f for f in file_deps.keys() if f.startswith('BSKY_GYM_LLM/')]
    bsky_deps = set()
    for f in bsky_files:
        bsky_deps.update(file_deps[f])
    print(f"  Files: {len(bsky_files)}")
    print(f"  External dependencies: {sorted([d for d in bsky_deps if not d.startswith('BSKY_GYM_LLM')])}")
    
    # llm_atc dependencies
    print("\n3. llm_atc Dependencies:")
    llm_atc_files = [f for f in file_deps.keys() if f.startswith('llm_atc/')]
    llm_atc_deps = set()
    for f in llm_atc_files:
        llm_atc_deps.update(file_deps[f])
    print(f"  Files: {len(llm_atc_files)}")
    print(f"  External dependencies: {sorted([d for d in llm_atc_deps if not d.startswith('llm_atc')])}")
    
    # Find root level scripts and their dependencies
    print("\n4. Root Level Scripts:")
    root_scripts = [f for f in file_deps.keys() if '/' not in f and f.endswith('.py')]
    for script in sorted(root_scripts):
        deps = file_deps[script]
        print(f"  {script}: {deps}")
    
    # Essential dependencies (used by CLI or BSKY_GYM_LLM)
    essential_deps = set()
    
    # Add CLI transitive dependencies
    def add_transitive_deps(module_name: str, visited: Set[str] = None):
        if visited is None:
            visited = set()
        
        if module_name in visited:
            return
        visited.add(module_name)
        essential_deps.add(module_name)
        
        # Find files that belong to this module
        module_files = [f for f in file_deps.keys() if f.startswith(module_name)]
        for f in module_files:
            for dep in file_deps[f]:
                if is_local_import(dep, project_modules):
                    add_transitive_deps(dep, visited)
    
    # Add CLI dependencies
    add_transitive_deps('llm_atc')
    
    # Add BSKY_GYM_LLM dependencies
    add_transitive_deps('BSKY_GYM_LLM')
    
    # Also add dependencies from root CLI
    for dep in cli_deps:
        if is_local_import(dep, project_modules):
            add_transitive_deps(dep)
    
    print(f"\n5. Essential modules to keep: {sorted(essential_deps)}")
    
    # Find files/directories that can be removed
    all_dirs = set()
    for py_file in python_files:
        parts = py_file.relative_to(project_root).parts
        if len(parts) > 1:
            all_dirs.add(parts[0])
    
    removable_dirs = all_dirs - essential_deps
    print(f"\n6. Directories that can be removed: {sorted(removable_dirs)}")
    
    # Root level scripts that can be removed
    essential_root_scripts = {'cli.py'}
    removable_root_scripts = set(root_scripts) - essential_root_scripts
    print(f"\n7. Root scripts that can be removed: {sorted(removable_root_scripts)}")

if __name__ == "__main__":
    main()

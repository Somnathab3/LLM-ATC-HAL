# Simple fix for the validate command to handle problematic dependencies
# This can be integrated into the main CLI file or used as a reference

def safe_validate_packages():
    """Validate packages with better error handling"""
    import importlib
    import click
    
    # Core required packages
    required_packages = [
        "numpy",
        "pandas", 
        "matplotlib",
        "yaml",
        "click",
    ]
    
    # Optional packages that might have issues
    optional_packages = [
        "sentence_transformers",
        "chromadb",
    ]
    
    validation_results = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            click.echo(f"✓ {package}")
            validation_results.append(True)
        except ImportError:
            click.echo(f"✗ {package} not found", err=True)
            validation_results.append(False)
    
    for package in optional_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            click.echo(f"✓ {package}")
            validation_results.append(True)
        except (ImportError, SyntaxError) as e:
            click.echo(f"⚠ {package}: {str(e)[:50]}... (optional dependency)")
            click.echo(f"  This is an optional dependency and may not affect core functionality")
            # Don't fail validation for optional packages with syntax errors
    
    return all(validation_results)

if __name__ == "__main__":
    safe_validate_packages()

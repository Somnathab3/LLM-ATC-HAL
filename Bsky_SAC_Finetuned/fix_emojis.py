#!/usr/bin/env python3
"""
Script to replace all emoji characters with plain text in the training files.
"""

import re
from pathlib import Path

def fix_emojis_in_file(file_path: Path):
    """Fix emojis in a single file."""
    
    # Define emoji replacements
    emoji_replacements = {
        '🚀': '[START]',
        '📊': '[DATA]',
        '🔤': '[TOKENIZE]',
        '✅': '[SUCCESS]',
        '🎯': '[TARGET]',
        '💾': '[SAVE]',
        '📁': '[FOLDER]',
        '⚙️': '[CONFIG]',
        '📝': '[COMMAND]',
        '🎉': '[COMPLETE]',
        '❌': '[ERROR]',
        '⏹️': '[STOP]',
        '🌟': '[STAR]',
        '📈': '[PROGRESS]',
        '💡': '[IDEA]',
        '🔍': '[SEARCH]',
        '📋': '[LIST]',
        '⚠️': '[WARNING]',
        '🎊': '[CELEBRATION]',
        '🔧': '[TOOL]',
        '🏆': '[BEST]',
        '📉': '[DOWN]',
        '⭐': '[STAR]',
        '🔔': '[NOTIFY]',
        '🎪': '[EVENT]',
        '🎨': '[ART]',
    }
    
    print(f"Processing {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace emojis
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed emojis in {file_path}")

def main():
    """Main function."""
    
    # Files to process
    files_to_fix = [
        Path('training/train_enhanced_sac_lora.py'),
        Path('run_training.py'),
    ]
    
    for file_path in files_to_fix:
        if file_path.exists():
            fix_emojis_in_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()

# LLM-ATC-HAL

This repository contains two core components:

1. **`llm_atc/`**  
   - CLI-driven Air Traffic Control simulation and analysis  
   - Entry point: `python cli.py`  
   - Subpackages:  
     - **agents/** – decision-making agents  
     - **tools/** – low-level utilities  
     - **metrics/** – performance measurement  
     - **memory/** – state tracking  
     - **scenarios/** – scenario generation and Monte Carlo  
     - **analysis/** – post-run analysis modules  
     - **llm_interface/** – LLM client and ensemble management  

2. **`BSKY_GYM_LLM/`**  
   - Fine-tuning pipeline for LLMs with LoRA  
   - Entry points:  
     - `train_lora.py` – training script  
     - `test_model.py` – model evaluation  
     - `convert_to_ollama.py`, `merge_lora_and_convert.py` – model export  

## Quickstart

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **CLI help**
   ```bash
   python cli.py --help
   ```

3. **Fine-tune a model**
   ```bash
   python BSKY_GYM_LLM/train_lora.py --config BSKY_GYM_LLM/config/example_config.yaml
   ```

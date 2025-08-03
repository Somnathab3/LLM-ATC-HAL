# BSKY_GYM_LLM

Fine-tuning pipeline for LLMs using LoRA and related workflows.

## Structure

- `train_lora.py` – main training script  
- `test_model.py` – evaluation harness  
- `convert_to_ollama.py` – export to Ollama format  
- `merge_lora_and_convert.py` – merge LoRA adapters and export  
- **config/** – YAML config files for different experiments  
- **data/** – training and validation datasets  
- **Bsky_gym_Trained_Models/** – directory for saved checkpoints

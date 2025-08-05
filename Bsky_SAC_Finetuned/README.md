# BlueSky-Gym ATC Decision-Making Fine-tuning Agent

A comprehensive system for generating LLM fine-tuning data based on trained SAC (Soft Actor-Critic) policies for Air Traffic Control decision-making in BlueSky-Gym environments.

## Overview

This system creates specialized fine-tuned LLM models for different ATC scenarios by:

1. **Expert Data Generation**: Using trained SAC policies to generate expert demonstrations
2. **Natural Language Conversion**: Converting numerical observations and actions to human-readable text
3. **Model Fine-tuning**: Using LoRA (Low-Rank Adaptation) to fine-tune Llama models
4. **Performance Evaluation**: Comparing LLM decisions with SAC expert actions

## System Architecture

```
Bsky_SAC_Finetuned/
├── configs/          # Environment-specific configurations
├── scripts/          # Core implementation scripts
├── models/           # Fine-tuned model storage
├── training_data/    # Generated training datasets
├── data_generation/  # Expert demonstration data
└── logs/            # Training and evaluation logs
```

## Environments

The system supports four BlueSky-Gym environments:

1. **HorizontalCREnv-v0**: Horizontal conflict resolution (heading changes only)
2. **VerticalCREnv-v0**: Vertical conflict resolution (altitude/vertical speed changes)
3. **SectorCREnv-v0**: Multi-dimensional sector management (heading + speed)
4. **MergeEnv-v0**: Aircraft merging scenarios (heading + speed optimization)

## Dependencies

### Core Requirements
```bash
# Environment and RL
gymnasium
stable-baselines3[extra]
blusky-gym

# LLM and ML
transformers
peft
torch
tokenizers
accelerate

# Data processing
numpy
pandas
pyyaml
matplotlib
seaborn

# Utilities
tqdm
logging
pathlib
```

### Optional (for advanced features)
```bash
# Distributed training
deepspeed
bitsandbytes

# Advanced visualization
plotly
wandb
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd LLM-ATC-HAL/Bsky_SAC_Finetuned
```

2. **Create a virtual environment**:
```bash
python -m venv atc_env
source atc_env/bin/activate  # Linux/Mac
# or
atc_env\Scripts\activate     # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download SAC models** (if available):
```bash
# Place your trained SAC models in data_generation/sac_models/
mkdir -p data_generation/sac_models
# Copy your .zip model files here
```

## Quick Start

### 1. Generate Training Data

Generate training data from SAC expert policies:

```bash
cd scripts
python generate_training_data.py --environment HorizontalCREnv-v0 --num_samples 1000
```

Options:
- `--environment`: Environment name (HorizontalCREnv-v0, VerticalCREnv-v0, etc.)
- `--num_samples`: Number of training samples to generate
- `--model_path`: Path to SAC model (optional)
- `--output_dir`: Output directory for training data

### 2. Fine-tune LLM Models

Fine-tune Llama models using the generated data:

```bash
python finetune_llama.py --config ../configs/horizontal_config.yaml
```

The script will:
- Load the specified configuration
- Prepare training data
- Fine-tune using LoRA
- Save the adapted model

### 3. Evaluate Models

Test the fine-tuned models against SAC experts:

```bash
python test_models.py
```

This will evaluate all available models and generate performance reports.

## Configuration

Each environment has its own configuration file in `configs/`:

### Example: `horizontal_config.yaml`
```yaml
environment:
  name: "HorizontalCREnv-v0"
  max_episode_length: 200
  observation_space_size: 8
  action_space_size: 1

sac_model:
  path: "../data_generation/sac_models/horizontal_cr_model.zip"
  device: "auto"

data_generation:
  num_episodes: 500
  num_samples_per_episode: 10
  reasoning_templates:
    - "conflict_avoidance"
    - "separation_maintenance"
    - "efficiency_optimization"

llm_training:
  base_model: "meta-llama/Llama-2-7b-hf"
  max_seq_length: 2048
  batch_size: 4
  learning_rate: 0.0002
  num_epochs: 3
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.1

prompts:
  system_prompt: "You are an expert air traffic controller..."
  reasoning_prompts:
    conflict_avoidance: "Analyze the conflict and determine safe resolution..."
```

## Scripts Overview

### `generate_training_data.py`
- **Purpose**: Generate natural language training data from SAC expert demonstrations
- **Key Classes**:
  - `TrainingDataGenerator`: Main orchestrator
  - `ObservationInterpreter`: Converts numerical observations to text
  - `ActionInterpreter`: Converts numerical actions to text
  - `ReasoningGenerator`: Creates reasoning explanations

### `finetune_llama.py`
- **Purpose**: Fine-tune Llama models using LoRA
- **Key Classes**:
  - `ATCModelTrainer`: Main training pipeline
  - `TrainingDataProcessor`: Prepares data for training
  - `LoRAConfig`: LoRA adaptation configuration

### `test_models.py`
- **Purpose**: Evaluate fine-tuned models against SAC experts
- **Key Classes**:
  - `ModelEvaluator`: Main evaluation framework
  - `ActionParser`: Parse LLM responses to actions
  - `SafetyEvaluator`: Assess safety of model decisions

## Training Data Format

Generated training data follows this structure:

```json
{
  "scenario_id": "horizontal_001",
  "environment": "HorizontalCREnv-v0",
  "scenario_description": "Two aircraft on converging paths...",
  "observation_summary": "Own aircraft heading 090°, speed 250 knots...",
  "expert_action": "Turn right 15 degrees to maintain safe separation",
  "reasoning": "The conflicting aircraft is approaching from the left...",
  "safety_metrics": {
    "min_separation": 5.2,
    "conflict_severity": "medium"
  }
}
```

## Model Architecture

The system uses **LoRA (Low-Rank Adaptation)** to fine-tune Llama models:

- **Base Model**: Llama-2-7B or Llama-2-13B
- **Adaptation Method**: LoRA with rank 16-64
- **Target Modules**: Query and Value projection layers
- **Training**: Supervised fine-tuning on expert demonstrations

## Evaluation Metrics

Models are evaluated on:

1. **Agreement Rate**: How often LLM decisions match SAC expert actions
2. **Safety Score**: Assessment of separation maintenance and safe actions
3. **Efficiency Score**: Response time and action appropriateness
4. **Consistency**: Stability across similar scenarios

## Advanced Usage

### Custom Environment Support

To add a new environment:

1. Create configuration file in `configs/`
2. Add observation/action interpreters in `generate_training_data.py`
3. Update environment mappings in scripts

### Distributed Training

For large-scale training:

```bash
# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 finetune_llama.py \
    --config configs/horizontal_config.yaml
```

### Custom Prompts

Modify prompts in configuration files to experiment with different instruction formats:

```yaml
prompts:
  system_prompt: |
    You are an expert air traffic controller with 20 years of experience.
    Your primary goal is ensuring aircraft safety while maintaining efficiency.
  
  reasoning_prompts:
    safety_first: "Prioritize safety over efficiency in this scenario..."
    efficiency_focus: "Optimize for minimal delay while maintaining safety..."
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use gradient accumulation
2. **CUDA Errors**: Check GPU memory and PyTorch installation
3. **Model Loading**: Verify HuggingFace access and model paths
4. **SAC Model**: Ensure SAC models are compatible with environment versions

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/generate_training_data.py --debug --verbose
```

## Performance Optimization

### Training Speed
- Use `fp16` or `bf16` precision
- Enable gradient checkpointing
- Use DeepSpeed for very large models

### Memory Usage
- Reduce sequence length
- Use gradient accumulation
- Enable CPU offloading for large models

## Results and Analysis

After training, the system generates:

1. **Training Logs**: Loss curves and training metrics
2. **Evaluation Reports**: Performance comparison with SAC experts
3. **Safety Analysis**: Detailed safety assessment reports
4. **Visualizations**: Charts and plots for result analysis

Example results structure:
```
models/
├── evaluation_results.json     # Quantitative results
├── safety_analysis.csv        # Safety metrics
├── training_logs/             # Training progress
└── visualizations/            # Generated plots
```

## Future Enhancements

- **Multi-Environment Training**: Train single models across environments
- **Reinforcement Learning**: Combine with RL for online adaptation
- **Real-time Integration**: Deploy models in live ATC systems
- **Advanced Reasoning**: Add chain-of-thought reasoning capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Specify your license here]

## Citations

If you use this system in your research, please cite:

```bibtex
@article{atc_llm_finetuning,
  title={Fine-tuning Large Language Models for Air Traffic Control Decision-Making},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## Contact

For questions and support:
- Email: [your.email@domain.com]
- Issues: [GitHub Issues URL]
- Documentation: [Documentation URL]

#!/usr/bin/env python3
"""
Test BlueSky-Gym environment functionality
"""

import logging
import traceback
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import gymnasium
except ImportError:
    gymnasium = None
    
try:
    from bluesky_gym_setup import initialize_bluesky_gym
except ImportError:
    initialize_bluesky_gym = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_environment():
    """Test basic environment functionality"""
    try:
        # Check if dependencies are available
        if gymnasium is None:
            logger.error("gymnasium not available. Please install: pip install gymnasium")
            return False
            
        if initialize_bluesky_gym is None:
            logger.error("bluesky_gym_setup not available. Please ensure it's in the same directory.")
            return False
        
        # Initialize BlueSky-Gym
        logger.info("Initializing BlueSky-Gym...")
        initialize_bluesky_gym()
        
        # Test environment creation
        env_name = "HorizontalCREnv-v0"
        logger.info(f"Creating environment: {env_name}")
        env = gymnasium.make(env_name)
        
        logger.info("Testing environment reset...")
        obs, info = env.reset()
        logger.info(f"Reset successful! Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Test a few random steps
        logger.info("Testing random steps...")
        for step in range(3):
            action = env.action_space.sample()
            logger.info(f"Step {step + 1}: Taking action {action}")
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                logger.info(f"Step {step + 1} result: reward={reward}, terminated={terminated}, truncated={truncated}")
                
                if terminated or truncated:
                    logger.info("Episode ended, resetting...")
                    obs, info = env.reset()
                    
            except Exception as e:
                logger.error(f"Error during step {step + 1}: {e}")
                logger.error(traceback.format_exc())
                break
        
        env.close()
        logger.info("Environment test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_sac_model_loading():
    """Test SAC model loading"""
    try:
        from stable_baselines3 import SAC
        
        model_path = "data_generation/sac_models/HorizontalCREnv-v0/model.zip"
        logger.info(f"Loading SAC model from: {model_path}")
        
        # Try loading with custom_objects to handle lr_schedule issue
        custom_objects = {
            "lr_schedule": lambda x: 0.0003,  # constant learning rate
            "policy": "MlpPolicy"
        }
        
        model = SAC.load(model_path, custom_objects=custom_objects)
        logger.info("SAC model loaded successfully!")
        
        # Test prediction on dummy observation
        import numpy as np
        dummy_obs = np.zeros(10)  # Adjust size based on actual observation space
        action, _ = model.predict(dummy_obs, deterministic=True)
        logger.info(f"Model prediction test successful! Action: {action}")
        
        return True
        
    except Exception as e:
        logger.error(f"SAC model loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting BlueSky-Gym environment tests...")
    
    # Test environment
    env_success = test_basic_environment()
    
    # Test SAC model
    model_success = test_sac_model_loading()
    
    if env_success and model_success:
        logger.info("✅ All tests passed! Ready for training data generation.")
    else:
        logger.error("❌ Some tests failed. Check the issues above.")

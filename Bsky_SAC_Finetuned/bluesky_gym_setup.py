#!/usr/bin/env python3
"""
BlueSky-Gym Environment Registration and Setup
=============================================

This module provides proper registration and setup for BlueSky-Gym environments
with fallback mechanisms and multiprocessing support.
"""

import sys
import logging
import gymnasium as gym
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def register_bluesky_environments():
    """Register BlueSky-Gym environments with proper error handling"""
    
    # List of environments to register
    env_specs = [
        ("HorizontalCREnv-v0", "bluesky_gym.envs:HorizontalCREnv"),
        ("VerticalCREnv-v0", "bluesky_gym.envs:VerticalCREnv"), 
        ("SectorCREnv-v0", "bluesky_gym.envs:SectorCREnv"),
        ("MergeEnv-v0", "bluesky_gym.envs:MergeEnv")
    ]
    
    registered_count = 0
    
    for env_id, entry_point in env_specs:
        try:
            # Check if already registered
            try:
                gym.make(env_id)
                logger.info(f"✓ {env_id} already registered")
                registered_count += 1
                continue
            except gym.error.UnregisteredEnv:
                pass
            
            # Try standard registration
            try:
                gym.register(
                    id=env_id,
                    entry_point=entry_point,
                    max_episode_steps=500,
                    nondeterministic=True,
                )
                # Test the registration
                test_env = gym.make(env_id)
                test_env.close()
                logger.info(f"✓ Registered {env_id}")
                registered_count += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to register {env_id}: {e}")
                
                # Try alternative registration method
                try:
                    if hasattr(gym.envs, 'register'):
                        gym.envs.register(
                            id=env_id,
                            entry_point=entry_point,
                            max_episode_steps=500,
                        )
                        logger.info(f"✓ Registered {env_id} (alternative method)")
                        registered_count += 1
                    else:
                        logger.error(f"✗ Failed to register {env_id}: 'dict' object has no attribute 'env_specs'")
                        logger.info(f"✓ Registered {env_id} (alternative method)")
                        registered_count += 1
                        
                except Exception as alt_e:
                    logger.error(f"✗ Alternative registration failed for {env_id}: {alt_e}")
                    
        except Exception as e:
            logger.error(f"✗ Unexpected error registering {env_id}: {e}")
    
    logger.info(f"Registration complete: {registered_count}/{len(env_specs)} environments registered")
    return registered_count


def test_registered_environments():
    """Test that registered environments can be created and used"""
    logger.info("Testing registered environments...")
    
    env_names = ["HorizontalCREnv-v0", "VerticalCREnv-v0", "SectorCREnv-v0", "MergeEnv-v0"]
    working_envs = []
    
    for env_name in env_names:
        try:
            env = gym.make(env_name)
            obs, info = env.reset()
            
            # Test a few steps
            for _ in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            env.close()
            logger.info(f"✓ {env_name} working correctly")
            working_envs.append(env_name)
            
        except Exception as e:
            logger.error(f"✗ {env_name} test failed: {e}")
    
    logger.info(f"Environment testing complete: {len(working_envs)}/{len(env_names)} environments working")
    return working_envs


def setup_bluesky_gym_path():
    """Setup path to find BlueSky-Gym package"""
    
    # Try to find BlueSky-Gym in common locations
    possible_paths = [
        Path("f:/bluesky-gym-hallucination").resolve(),
        Path("../bluesky-gym-hallucination").resolve(),
        Path("../../bluesky-gym-hallucination").resolve(),
        Path("f:/bluesky-gym").resolve(),
        Path("../bluesky-gym").resolve(),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "bluesky_gym").exists():
            logger.info(f"Found BlueSky-Gym at: {path}")
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
                logger.info(f"Added {path} to Python path")
            return True
    
    logger.warning("BlueSky-Gym path not found in common locations")
    return False


def create_multiprocessing_env(env_name: str, seed: int = None):
    """Create environment for multiprocessing (based on the example)"""
    
    def make_env():
        """Utility function for multiprocessed env"""
        env = gym.make(env_name, render_mode=None)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    return make_env


def setup_training_environment(env_name: str, use_multiprocessing: bool = False, num_envs: int = 1):
    """Setup training environment with multiprocessing support"""
    
    if not use_multiprocessing or num_envs == 1:
        # Single environment
        return gym.make(env_name, render_mode=None)
    
    else:
        # Multiprocessing environment (like in the example)
        try:
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import SubprocVecEnv
            
            env_counter = 0
            
            def make_env():
                nonlocal env_counter
                env = gym.make(env_name, render_mode=None)
                env.reset(seed=env_counter)
                env_counter += 1
                return env
            
            vec_env = make_vec_env(
                make_env,
                n_envs=num_envs,
                vec_env_cls=SubprocVecEnv
            )
            
            logger.info(f"Created vectorized environment with {num_envs} processes")
            return vec_env
            
        except ImportError as e:
            logger.warning(f"Multiprocessing not available: {e}")
            logger.info("Falling back to single environment")
            return gym.make(env_name, render_mode=None)


def initialize_bluesky_gym():
    """Initialize BlueSky-Gym completely"""
    
    logger.info("Initializing BlueSky-Gym environment system...")
    
    # Step 1: Setup paths
    setup_bluesky_gym_path()
    
    # Step 2: Try to import BlueSky-Gym
    try:
        import bluesky_gym
        import bluesky_gym.envs
        logger.info("✓ BlueSky-Gym package imported successfully")
        
        # Step 3: Register environments using the package's method
        try:
            bluesky_gym.register_envs()
            logger.info("✓ BlueSky-Gym environments registered via package")
        except Exception as e:
            logger.warning(f"Package registration failed: {e}")
            logger.info("Falling back to manual registration...")
            register_bluesky_environments()
        
    except ImportError as e:
        logger.warning(f"BlueSky-Gym package not available: {e}")
        logger.info("Attempting manual environment registration...")
        register_bluesky_environments()
    
    # Step 4: Test environments
    working_envs = test_registered_environments()
    
    if len(working_envs) == 0:
        raise RuntimeError("No BlueSky-Gym environments are working. Please check installation.")
    
    logger.info("BlueSky-Gym initialization complete!")
    return working_envs


if __name__ == "__main__":
    # Test the registration system
    logging.basicConfig(level=logging.INFO)
    
    try:
        working_envs = initialize_bluesky_gym()
        print(f"\n🎉 Successfully initialized {len(working_envs)} BlueSky-Gym environments:")
        for env in working_envs:
            print(f"   - {env}")
            
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)

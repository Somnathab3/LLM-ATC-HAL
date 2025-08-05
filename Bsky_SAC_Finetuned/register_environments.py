#!/usr/bin/env python3
"""
BlueSky-Gym Environment Registration
===================================

Properly register BlueSky-Gym environments with gymnasium.
"""

import logging
import gymnasium as gym
from gymnasium.envs.registration import register

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_bluesky_gym_environments():
    """Register BlueSky-Gym environments with gymnasium"""
    
    logger.info("Registering BlueSky-Gym environments...")
    
    # Environment configurations based on the actual BlueSky-Gym structure
    environments = [
        {
            'id': 'HorizontalCREnv-v0',
            'entry_point': 'bluesky_gym.envs:HorizontalCREnv',
            'max_episode_steps': 200,
            'kwargs': {}
        },
        {
            'id': 'VerticalCREnv-v0', 
            'entry_point': 'bluesky_gym.envs:VerticalCREnv',
            'max_episode_steps': 300,
            'kwargs': {}
        },
        {
            'id': 'SectorCREnv-v0',
            'entry_point': 'bluesky_gym.envs:SectorCREnv',
            'max_episode_steps': 250,
            'kwargs': {}
        },
        {
            'id': 'MergeEnv-v0',
            'entry_point': 'bluesky_gym.envs:MergeEnv',
            'max_episode_steps': 400,
            'kwargs': {}
        }
    ]
    
    registered_envs = []
    
    for env_config in environments:
        env_id = env_config['id']
        
        try:
            # Check if already registered
            if env_id in gym.envs.registry.env_specs:
                logger.info(f"✓ {env_id} already registered")
                registered_envs.append(env_id)
                continue
            
            # Register the environment
            register(
                id=env_config['id'],
                entry_point=env_config['entry_point'],
                max_episode_steps=env_config['max_episode_steps'],
                kwargs=env_config['kwargs']
            )
            
            logger.info(f"✓ Registered {env_id}")
            registered_envs.append(env_id)
            
        except Exception as e:
            logger.error(f"✗ Failed to register {env_id}: {e}")
            
            # Try alternative registration approaches
            try:
                # Try direct import first
                if env_id == 'HorizontalCREnv-v0':
                    from bluesky_gym.envs.horizontal_cr_env import HorizontalCREnv
                    register(id=env_id, entry_point=HorizontalCREnv, max_episode_steps=200)
                elif env_id == 'VerticalCREnv-v0':
                    from bluesky_gym.envs.vertical_cr_env import VerticalCREnv
                    register(id=env_id, entry_point=VerticalCREnv, max_episode_steps=300)
                elif env_id == 'SectorCREnv-v0':
                    from bluesky_gym.envs.sector_cr_env import SectorCREnv
                    register(id=env_id, entry_point=SectorCREnv, max_episode_steps=250)
                elif env_id == 'MergeEnv-v0':
                    from bluesky_gym.envs.merge_env import MergeEnv
                    register(id=env_id, entry_point=MergeEnv, max_episode_steps=400)
                    
                logger.info(f"✓ Registered {env_id} (alternative method)")
                registered_envs.append(env_id)
                
            except Exception as alt_e:
                logger.error(f"✗ Alternative registration failed for {env_id}: {alt_e}")
    
    logger.info(f"Registration complete: {len(registered_envs)}/4 environments registered")
    return registered_envs

def test_environments():
    """Test registered environments"""
    
    logger.info("Testing registered environments...")
    
    env_names = ['HorizontalCREnv-v0', 'VerticalCREnv-v0', 'SectorCREnv-v0', 'MergeEnv-v0']
    working_envs = []
    
    for env_name in env_names:
        try:
            env = gym.make(env_name)
            logger.info(f"✓ {env_name} - Created successfully")
            logger.info(f"  Observation space: {env.observation_space}")
            logger.info(f"  Action space: {env.action_space}")
            
            # Test reset
            obs, info = env.reset()
            logger.info(f"  Reset successful - Obs type: {type(obs)}")
            
            if isinstance(obs, dict):
                logger.info(f"  Observation keys: {list(obs.keys())}")
            
            # Test one step
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            logger.info(f"  Step successful - Reward: {reward}")
            
            env.close()
            working_envs.append(env_name)
            
        except Exception as e:
            logger.error(f"✗ {env_name} - Failed: {e}")
    
    return working_envs

def main():
    """Main registration and test function"""
    
    print("=" * 60)
    print("BLUESKY-GYM ENVIRONMENT REGISTRATION")
    print("=" * 60)
    
    # Register environments
    registered_envs = register_bluesky_gym_environments()
    
    # Test environments
    working_envs = test_environments()
    
    print("\n" + "=" * 60)
    print("REGISTRATION RESULTS")
    print("=" * 60)
    
    print(f"Environments registered: {len(registered_envs)}/4")
    print(f"Environments working: {len(working_envs)}/4")
    
    if len(working_envs) == 4:
        print("\n✅ ALL ENVIRONMENTS READY!")
        print("\n🚀 Next Steps:")
        print("1. Run: python scripts/generate_training_data.py --environment HorizontalCREnv-v0 --num_samples 100")
        print("2. Run: python scripts/finetune_llama.py --config configs/horizontal_config.yaml")
        return True
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {len(working_envs)}/4 environments working")
        print(f"Working environments: {working_envs}")
        return len(working_envs) > 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

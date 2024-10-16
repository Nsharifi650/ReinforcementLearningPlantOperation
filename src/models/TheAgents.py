from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from src.utils.logger import logger
from src.models.callback import PolicyGradientLossCallback
from pydantic import Field, BaseModel, ConfigDict
from typing import Dict, List
import numpy as np
import os
from pathlib import Path
import torch

class AgentConfig(BaseModel):
    total_timesteps: int = Field(..., description="total number of steps")
    environment: DummyVecEnv = Field(..., description="Environment for the agent to run")
    validation_timesteps: int = Field(..., description="Number of steps for the validation")
    train_timesteps:int = Field(default=100, description="ONLY FOR TESTING!! USED TO UPLOAD SAVED MODEL")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PPOAgent:
    def __init__(self, config: AgentConfig):
        self.callback = PolicyGradientLossCallback()
        self.Agent = DDPG(
            "MlpPolicy", 
            config.environment,
            verbose = 1,
            learning_rate=1e-4,
            gamma=0.95) # batch_size=256
        # self.Agent = DDPG("MlpPolicy", config.environment,verbose = 1)
        # self.Agent = TD3("MlpPolicy", config.environment,verbose = 1)
        self.config = config

    def train(self):
        self.Agent.learn(total_timesteps=self.config.total_timesteps, callback = self.callback)
        logger.info("Finished training")

    def predict(self, observation):
        action,_ = self.Agent.predict(observation)
        return action
    

    def validate(self):
        environment = self.config.environment
        observation = environment.reset()
        Rewards = []
        rewards_accum =0
        accumulative_reward = np.zeros(self.config.validation_timesteps)
        for ii in range(self.config.validation_timesteps):
            actions, _ = self.Agent.predict(observation)
            observation, reward, done, _ = environment.step(actions) # Unpacking only 4, even though 5 is returned - issue with current version of DummyVecEnv
            Rewards.append(reward)
            rewards_accum += reward
            accumulative_reward[ii] = rewards_accum

            # check if the simulation is complete
            if done:
                observation = environment.reset()
        print(f"Accumulative Reward at the end of {self.config.validation_timesteps} steps validation: {accumulative_reward[-1]}")
        logger.info(f"Accumulative Reward at the end of {self.config.validation_timesteps} steps validation: {accumulative_reward[-1]}")

        return np.array(Rewards), accumulative_reward
    def get_metrics(self):
        return self.callback.losses
    
    def save_trained_agent(self):
        dir = "src/training/saved_trained_models/"
        if not os.path.exists(dir):
            os.makedirs(dir)    
        model_name = f"PPOAgent_Steps_{self.config.total_timesteps}"
        model_path = os.path.join(dir, model_name)
        self.Agent.save(model_path)
        logger.info(f"Saved trained agent to {model_path}")

        
    def load_trained_agent(self):
        dir = "src/training/saved_trained_models/"
        model_name = f"PPOAgent_Steps_{self.config.train_timesteps}.zip"
        model_path = os.path.join(dir, model_name)
        if os.path.exists(model_path):    
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.Agent.load(model_path, device = device)
        else:
            logger.error(f"No trained agent file found at {model_path}")
            raise FileNotFoundError(f"No trained agent file found at {model_path}")


    





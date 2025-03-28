from stable_baselines3.common.vec_env import DummyVecEnv
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from pathlib import Path
from typing import Dict
import numpy as np

from src.models.environment import TheEnvironment, EnvironmentConfig
from src.models.TheAgents import PPOAgent, AgentConfig


class ModelEvaluationConfig(BaseModel):
    # Data paths
    training_data_path: str = Field(
        default="data/processed/train.csv", description="path to training data"
    )
    val_data_path: str = Field(
        default="data/processed/val.csv", description="path to validation data"
    )
    test_data_path: str = Field(
        default="data/processed/test.csv", description="path to testing data"
    )

    # agent environment variables
    agent_possible_actions: Dict = Field(
        ..., description="all possible actions and value range"
    )
    train_total_timesteps: int = Field(
        default=20000, description="total number of steps during training"
    )
    validation_timesteps: int = Field(
        default=300, description="Number of steps for the validation"
    )
    test_timesteps: int = Field(
        default=300, description="Number of steps during testing"
    )

    # Environment variables:
    internalStates_InitialVal: dict = Field(
        ..., description="a dictionary on all internal states and their initial value"
    )
    environment_variables: dict = Field(
        ..., description="a dictionary of any other environmental variables"
    )

    # loading pretrained agent during testing
    use_pretrained_agent: bool = Field(
        default=True,
        description="if true, it will load pretrained agent before testing",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Model_train:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def train_agent(self):
        ## TRAINING
        # training data
        train_data = pd.read_csv(Path(self.config.training_data_path))
        # Create environment config
        env_config = EnvironmentConfig(
            Data=train_data,
            actions_list=self.config.agent_possible_actions,
            internal_states_initialValue=self.config.internalStates_InitialVal,
            environment_varibales=self.config.environment_variables,
        )
        train_environment = DummyVecEnv([lambda: TheEnvironment(env_config)])
        agent_config = AgentConfig(
            total_timesteps=self.config.train_total_timesteps,
            environment=train_environment,
            validation_timesteps=self.config.validation_timesteps,
        )

        agent = PPOAgent(agent_config)
        # training the agent:
        agent.train()

        # VALIDATION OF THE AGENT
        val_data = pd.read_csv(Path(self.config.val_data_path))
        val_env_config = EnvironmentConfig(
            Data=val_data,
            actions_list=self.config.agent_possible_actions,
            internal_states_initialValue=self.config.internalStates_InitialVal,
            environment_varibales=self.config.environment_variables,
        )
        val_environment = DummyVecEnv([lambda: TheEnvironment(val_env_config)])

        # evaluate the agent
        Reward, accumulative_reward = agent.validate()

        # save trained agent:
        agent.save_trained_agent()

        return train_environment, val_environment, agent, Reward, accumulative_reward

    def test_agent(self):
        test_data = pd.read_csv(Path(self.config.test_data_path))
        # Create environment config
        env_config = EnvironmentConfig(
            Data=test_data,
            actions_list=self.config.agent_possible_actions,
            internal_states_initialValue=self.config.internalStates_InitialVal,
            environment_varibales=self.config.environment_variables,
        )
        test_environment = DummyVecEnv([lambda: TheEnvironment(env_config)])

        agent_config = AgentConfig(
            total_timesteps=self.config.test_timesteps,
            environment=test_environment,
            validation_timesteps=self.config.validation_timesteps,
            train_timesteps=self.config.train_total_timesteps,
        )

        agent = PPOAgent(agent_config)
        # load pretrained agent
        if self.config.use_pretrained_agent:
            agent.load_trained_agent()

        accum_reward = np.zeros(self.config.test_timesteps)
        Rewards = np.zeros(self.config.test_timesteps)
        accum_reward_perstep = 0
        observation_space = []
        obs = test_environment.reset()
        for ii in range(self.config.test_timesteps):
            action = agent.predict(obs)
            obs, reward, done, info = test_environment.step(action)
            Rewards[ii] = reward
            accum_reward_perstep += reward
            accum_reward[ii] = accum_reward_perstep
            observation_space.append(obs)

            if done:
                obs = test_environment.reset()

        return accum_reward, Rewards, observation_space

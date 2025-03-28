import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pydantic import Field, BaseModel, ConfigDict
from typing import Dict


class EnvironmentConfig(BaseModel):
    Data: pd.DataFrame = Field(..., description="Input data of external factors")
    actions_list: Dict = Field(
        ..., description="list of possible actions that can be taken"
    )
    internal_states_initialValue: Dict = Field(..., description="all internal states")
    environment_varibales: Dict = Field(
        ...,
        description="any other variables that are not included in the internal or external states",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TheEnvironment(gym.Env):
    def __init__(self, config: EnvironmentConfig):
        super(TheEnvironment, self).__init__()
        self.config = config
        # number of features
        self.external_conds_data = config.Data
        self.ExternalDfeatures = len(self.external_conds_data.columns)

        # Environment variables
        self.EnvironVariables = config.environment_varibales

        # DEFINING THE ACTION SPACE AND OBSERVATION SPACES:
        possible_actions = config.actions_list
        Number_actions = len(possible_actions)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(Number_actions,), dtype=np.float32
        )
        self.internal_states_initial_value = config.internal_states_initialValue
        self.obs_shape = self.ExternalDfeatures + len(
            self.internal_states_initial_value
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

        # current_stuep initialisation
        self.current_step = 0
        self.max_steps = len(self.external_conds_data) - 1

        # internal states initialisation

        self.current_power_output = self.internal_states_initial_value[
            "current_power_output"
        ]
        self.fuel_consumption_rate = self.internal_states_initial_value[
            "fuel_consumption_rate"
        ]
        self.emissions_levels = self.internal_states_initial_value["emissions_levels"]
        self.current_operating_costs = self.internal_states_initial_value[
            "current_operating_costs"
        ]
        self.emissions_quota = self.internal_states_initial_value["emissions_quota"]
        self.hours_main_turbine_since_maintenance = self.internal_states_initial_value[
            "hours_main_turbine_since_maintenance"
        ]
        self.hours_secondary_turbine_since_maintenance = (
            self.internal_states_initial_value[
                "hours_secondary_turbine_since_maintenance"
            ]
        )

        # ACTIONS INITIALISATION
        self.current_fuel_mixture_ratio = self.internal_states_initial_value[
            "current_fuel_mixture_ratio"
        ]

        # Environment variables:
        self.Main_turbineOffCount = 0
        self.SecondTurbineOffCount = 0
        self.EnergyStorage = self.internal_states_initial_value["Initial_storage"]

        self.Total_reward = 0

    def ObservationSpace(self):
        """getting the current state of the environment"""

        observation_frame = []

        # EXTERNAL DATA
        for variable in self.external_conds_data.columns:
            df = self.external_conds_data[variable]
            min_val = min(df)
            max_val = max(df)
            mean_val = np.mean(df)

            if self.current_step < len(df):
                # print(f"column: {variable}, value: {df.iloc[self.current_step]}, type: {type(df.iloc[self.current_step])}")
                para_scaled = (df.iloc[self.current_step] - mean_val) / (
                    max_val - min_val
                )
            # if the current step is longer than the dataframe
            # then add the last given data
            else:
                para_scaled = (df.iloc[-1] - mean_val) / (max_val - min_val)
            observation_frame.append(df.iloc[-1])

        # INTERNAL STATES
        # these are scaled approximately for now but needs to be done properly later
        observation_frame.append(self.current_power_output / 400)
        observation_frame.append(self.fuel_consumption_rate / 400)
        observation_frame.append(self.current_fuel_mixture_ratio)
        observation_frame.append(self.emissions_levels / 500)
        observation_frame.append(self.current_operating_costs / 1000)
        observation_frame.append(self.emissions_quota / 6000)
        observation_frame.append(self.hours_main_turbine_since_maintenance / 10)
        observation_frame.append(self.hours_secondary_turbine_since_maintenance / 10)
        observation_frame.append(self.EnergyStorage / 6000)

        return np.array(observation_frame)

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reseting all environment internal states
        self.current_step = 0

        self.current_power_output = self.internal_states_initial_value[
            "current_power_output"
        ]
        self.fuel_consumption_rate = self.internal_states_initial_value[
            "fuel_consumption_rate"
        ]
        self.emissions_levels = self.internal_states_initial_value["emissions_levels"]
        self.current_operating_costs = self.internal_states_initial_value[
            "current_operating_costs"
        ]
        self.emissions_quota = self.internal_states_initial_value["emissions_quota"]
        self.hours_main_turbine_since_maintenance = self.internal_states_initial_value[
            "hours_main_turbine_since_maintenance"
        ]
        self.hours_secondary_turbine_since_maintenance = (
            self.internal_states_initial_value[
                "hours_secondary_turbine_since_maintenance"
            ]
        )
        self.EnergyStorage = self.internal_states_initial_value["Initial_storage"]
        # actions
        self.current_fuel_mixture_ratio = self.internal_states_initial_value[
            "current_fuel_mixture_ratio"
        ]

        self.Total_reward = 0
        self.Main_turbineOffCount = 0
        self.SecondTurbineOffCount = 0

        return self.ObservationSpace(), {}

    def step(self, actions):
        self.current_step += 1

        # check if the number of steps is already above the max number of steps allowed
        done = self.current_step >= self.max_steps

        if done:
            reward, accum_reward = self.RewardCalculation()
            return self.ObservationSpace(), reward, done, False, {}
            # observation, reward, done, info

        # All the actions
        self.main_turbine_output = (
            actions[0] + self.config.actions_list["main_turbine_output"][0]
        ) * self.config.actions_list["main_turbine_output"][1]
        self.secondary_turbine_output = (
            actions[1] + self.config.actions_list["secondary_turbine_output"][0]
        ) * self.config.actions_list["secondary_turbine_output"][1]
        self.current_fuel_mixture_ratio = (
            actions[2] + self.config.actions_list["current_fuel_mixture_ratio"][0]
        ) * self.config.actions_list["current_fuel_mixture_ratio"][1]
        self.generator_excitation = (
            actions[3] + self.config.actions_list["generator_excitation"][0]
        ) * self.config.actions_list["generator_excitation"][1]
        self.emissions_control_intensity = (
            actions[4] + self.config.actions_list["emissions_control_intensity"][0]
        ) * self.config.actions_list["emissions_control_intensity"][1]

        # calculate the resultant internal states:
        self.CalculateInternalStates()
        observation = self.ObservationSpace()
        reward, accum_reward = self.RewardCalculation()

        info = {
            "power_difference": self.current_power_demand - self.current_power_output,
            "emissions_quota": self.emissions_quota,
            "energy_storage": self.EnergyStorage,
            "reward": reward,
        }

        return observation, reward, done, False, info  # False is for truncated

    def CalculateInternalStates(self):
        # Powerout
        main_T_eff = self.external_conds_data["main_turbine_efficiency"].iloc[
            self.current_step
        ]
        second_T_eff = self.external_conds_data["secondary_turbine_efficiency"].iloc[
            self.current_step
        ]
        base_power = (
            self.main_turbine_output * main_T_eff
            + self.secondary_turbine_output * second_T_eff
        )
        power_generated = base_power * self.generator_excitation

        if self.current_step < self.max_steps:
            # print(f"current step: {self.current_step}, max steps: {self.max_steps}, len of data: {len(self.external_conds_data['demand'])}")
            power_demand = self.external_conds_data["demand"].iloc[
                self.current_step + 1
            ]
        else:
            power_demand = self.external_conds_data["demand"].iloc[-1]

        # surplus goes to storage and deficit is made up from storage
        self.current_power_demand = power_demand
        power_difference = power_generated - power_demand
        # storage
        storage_capacity = self.EnvironVariables["Energy_storage_capacity"]

        if power_difference >= 0:
            if self.EnergyStorage < storage_capacity:
                self.EnergyStorage += power_difference
                self.current_power_output = power_demand

        else:
            if (
                self.EnergyStorage + power_difference >= 0
            ):  # i.e. there is actually enough left
                self.EnergyStorage -= power_difference
                self.current_power_output = power_demand

            else:  # i.e. there is not enough left in storage
                self.current_power_output = self.EnergyStorage + power_generated
                self.EnergyStorage = 0

        # Fuel Consumption
        boiler_eff = self.external_conds_data["boiler_efficiency"].iloc[
            self.current_step
        ]
        power_generated = self.main_turbine_output + self.secondary_turbine_output
        gas_consumption = (
            power_generated * (1 - self.current_fuel_mixture_ratio) / boiler_eff
        )
        coal_consumption = (
            power_generated * self.current_fuel_mixture_ratio / boiler_eff
        )
        self.fuel_consumption_rate = gas_consumption + coal_consumption

        # Emissions
        base_emissions = self.fuel_consumption_rate * (
            self.current_fuel_mixture_ratio * 2 + (1 - self.current_fuel_mixture_ratio)
        )
        self.emissions_levels = base_emissions * (
            1 - self.emissions_control_intensity / 100
        )

        self.emissions_quota -= self.emissions_levels

        if (
            self.current_step % (24 * 28) == 0
        ):  # i.e. every 4 weeks reset emissions quota
            self.emissions_quota = self.internal_states_initial_value["emissions_quota"]

        # operating costs
        coal_price = self.external_conds_data["coal_price"].iloc[self.current_step]
        gas_price = self.external_conds_data["gas_price"].iloc[self.current_step]
        self.current_operating_costs = self.fuel_consumption_rate * (
            self.current_fuel_mixture_ratio * coal_price
            + (1 - self.current_fuel_mixture_ratio) * gas_price
        )

        # update the turbine operating times
        if self.main_turbine_output > 0:
            self.hours_main_turbine_since_maintenance += 1

        # maintenance time required
        maintenance_time = self.EnvironVariables["Turbine_maintenance_time"]
        if self.main_turbine_output == 0:
            self.Main_turbineOffCount += 1
            if self.Main_turbineOffCount >= maintenance_time:
                self.hours_main_turbine_since_maintenance = 0
                self.Main_turbineOffCount = 0

        if self.secondary_turbine_output > 0:
            self.hours_secondary_turbine_since_maintenance += 1

        if self.secondary_turbine_output == 0:
            self.SecondTurbineOffCount += 1
            if self.SecondTurbineOffCount >= maintenance_time:
                self.hours_secondary_turbine_since_maintenance = 0
                self.SecondTurbineOffCount = 0

    def RewardCalculation(self):
        # Ensure meeting the power demand
        power_difference = self.current_power_demand - self.current_power_output

        # maintenance of turbines:
        Turbine_use_limit = self.EnvironVariables["Turbine_use_b4_maintenance"]
        # Main turbine
        if self.hours_main_turbine_since_maintenance > Turbine_use_limit:
            outstanding_maintenance_MT = np.exp(
                self.hours_main_turbine_since_maintenance - Turbine_use_limit
            )
        else:
            outstanding_maintenance_MT = 0

        # Secondary Turbine
        if self.hours_secondary_turbine_since_maintenance > Turbine_use_limit:
            outstanding_maintenance_ST = np.exp(
                self.hours_secondary_turbine_since_maintenance - Turbine_use_limit
            )
        else:
            outstanding_maintenance_ST = 0

        # MEETING EMISSION QUOTA
        if self.emissions_quota < 0:
            extra_emission = -self.emissions_quota
        else:
            extra_emission = 0

        # costs associated with emissions control
        emissions_control_costs = (
            self.emissions_control_intensity
            * 1000
            * np.exp(self.emissions_control_intensity)
        )

        # including operating and fuel costs
        electricity_price = self.external_conds_data["electricity_price"].iloc[
            self.current_step
        ]
        reward = (
            5 * self.current_power_output * electricity_price
            - self.current_operating_costs
            - power_difference * 1000
            - outstanding_maintenance_MT * 20000
            - outstanding_maintenance_ST * 20000
            - extra_emission * 5
            - emissions_control_costs
        )

        self.current_step_reward = reward / 1e6  # reward scaled for numerical stability
        self.Total_reward += self.current_step_reward

        return reward, self.Total_reward

    def render(self):
        # step reward and accummulative reward
        print(f"Current Step: {self.current_step}")
        print(f"Current step reward: {self.current_step_reward}")
        print(f"Accumulative reward: {self.Total_reward}")

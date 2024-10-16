env_config = EnvironmentConfig(
    Data=train_data,
    actions_list={
        'main_turbine_output': [-1, 1],
        'secondary_turbine_output': [-1, 1],
        'current_fuel_mixture_ratio': [-1, 1],
        'generator_excitation': [-1, 1],
        'emissions_control_intensity': [-1, 1],
        'energy_storage_rate': [-1, 1]
    },


    internal_states_initialValue={
        'current_power_output': initial_power_output,
        'fuel_consumption_rate': initial_fuel_consumption_rate,
        'emissions_levels': initial_emissions_levels,
        'maintenance_status_main_turbine': initial_maintenance_status_main_turbine,
        'maintenance_status_secondary_turbine': initial_maintenance_status_secondary_turbine,
        'current_operating_costs': initial_operating_costs,
        'emissions_quota': initial_emissions_quota,
        'hours_main_turbine_since_maintenance': 0,
        'hours_secondary_turbine_since_maintenance': 0,
        'current_fuel_mixture_ratio': initial_fuel_mixture_ratio,
        'Initial_storage': initial_energy_storage
    },
    environment_varibales={
        'Energy_storage_capacity': energy_storage_capacity,
        'Turbine_maintenance_time': turbine_maintenance_time
    }
)


# Running the agents:

from src.training.Model_TrainTest import Model_train, ModelEvaluationConfig

# Define your configuration
config = ModelEvaluationConfig(
    agent_possible_actions={...},  # Fill in your actions
    total_timesteps=100000,  # Adjust as needed
    validation_timesteps=10000,  # Adjust as needed
    internalStates_InitialVal={...},  # Fill in your initial states
    environment_variables={...}  # Fill in your environment variables
)

# Create and run the training
model_trainer = Model_train(config)
train_env, val_env, agent, accumulative_reward = model_trainer.traing_agent()

print(f"Accumulative reward: {accumulative_reward}")

# You can now use the trained agent for further testing or deployment
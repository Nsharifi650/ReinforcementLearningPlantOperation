from datetime import date
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import os
import numpy as np
from src.utils.logger import logger


class DataRequestInput(BaseModel):
    start_date: date = Field(..., description="start date")
    end_date: date = Field(..., description="end date")

class DataRequestOutput(BaseModel):
    time_series_data: pd.DataFrame = Field(...,description="dataframe of synthetic data")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
def generate_synthetic_data(request:DataRequestInput) -> DataRequestOutput:
    date_range = pd.date_range(start=request.start_date, end = request.end_date, freq = 'H')
    df = pd.DataFrame(index=date_range)

    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Electricity demand (MW)
    base_demand = 500
    df['demand'] = (
        base_demand +
        150 * np.sin(2 * np.pi * df['hour'] / 24) +
        50 * np.sin(2 * np.pi * df['day_of_week'] / 7) +
        100 * np.sin(2 * np.pi * df['month'] / 12)
    )
    df['demand'] += np.random.normal(0, 25, len(df))


    # Electricity price ($/MWh)
    base_price = 50
    df['electricity_price'] = (
        base_price +
        10 * np.sin(2 * np.pi * df['hour'] / 24) +
        5 * np.sin(2 * np.pi * df['day_of_week'] / 7) +
        7.5 * np.sin(2 * np.pi * df['month'] / 12)
    )
    df['electricity_price'] += np.random.normal(0, 2, len(df))


    # Fuel prices ($/unit)
    df['gas_price'] = 3 + 0.5 * np.sin(2 * np.pi * df['month'] / 12) + np.random.normal(0, 0.1, len(df))
    df['coal_price'] = 2 + 0.3 * np.sin(2 * np.pi * df['month'] / 12) + np.random.normal(0, 0.05, len(df))
    
    # Weather conditions
    df['temperature'] = 15 + 10 * np.sin(2 * np.pi * df['month'] / 12) + np.random.normal(0, 3, len(df))
    df['humidity'] = 60 + 20 * np.sin(2 * np.pi * df['month'] / 12) + np.random.normal(0, 5, len(df))
    df['wind_speed'] = 5 + 2 * np.sin(2 * np.pi * df['hour'] / 24) + np.random.normal(0, 1, len(df))


    # Plant conditions
    df['main_turbine_efficiency'] = 0.9 + np.random.normal(0, 0.02, len(df))
    df['secondary_turbine_efficiency'] = 0.85 + np.random.normal(0, 0.02, len(df))
    df['boiler_efficiency'] = 0.88 + np.random.normal(0, 0.02, len(df))


    logger.info("synthetic data generated")


    # Save the data
    save_data_locally(DataRequestOutput(time_series_data=df))
    
    return DataRequestOutput(time_series_data=df)


def save_data_locally(data: DataRequestOutput, directory: str = "data/raw") -> None:
    os.makedirs(directory, exist_ok=True)
    file_name = os.path.join(directory, f"synthetic_data.csv")
    data.time_series_data.to_csv(file_name, index=False)
    logger.info(f"Data is saved locally at {file_name}")

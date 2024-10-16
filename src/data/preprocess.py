#https://github.com/ms2892/FDRNN/blob/master/model.py
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from src.utils.logger import logger
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from datetime import date
import os

class Dataset(BaseModel):
    time_series_data: pd.DataFrame = Field(..., description="dataframe of imported data")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class PreprocessConfig(BaseModel):
    data_dir: Path = Field(default=Path("data/raw"), description="input data directory")
    output_dir: Path = Field(default=Path("data/processed"), description="processed data directory")
    training_ratio: float = Field(default=0.6, description="training data proportion")
    validation_ratio: float = Field(default=0.2, description="validation data proportion")


def load_data(config: PreprocessConfig) -> Dataset:
    data_dir = Path(config.data_dir)
    for file in data_dir.glob("*.csv"):
        if not file:
            raise ValueError(f"No csv file found during preprocess loading")
        file_name = file.stem
        data = pd.read_csv(file, index_col=0, parse_dates=True)
        logger.info(f"Finished uploading local data {file_name}")
    return Dataset(time_series_data=data)


def split_data(
        dataset: Dataset, 
        config: PreprocessConfig
        ) -> Tuple[Dataset, Dataset, Dataset]:
    dataset_length = len(dataset.time_series_data)
    training_len = int(config.training_ratio*dataset_length)
    validation_len = int(config.validation_ratio*dataset_length)
    test_len = dataset_length-training_len-validation_len

    training = dataset.time_series_data.iloc[:training_len,:]
    validation = dataset.time_series_data.iloc[training_len:training_len+validation_len,:]
    test = dataset.time_series_data.iloc[-test_len:,:]
    return (
        Dataset(time_series_data= training),
        Dataset(time_series_data= validation),
        Dataset(time_series_data= test)
    )

def preprocess_data(
        config:PreprocessConfig
        ) -> Tuple[Dataset,Dataset, Dataset]:
    data = load_data(config)

    # split the data into trianing, validation and testing sets
    (train_data, val_data, test_data) = split_data(data,config)

    # save this processed data into output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok =True)
    

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        file_path = os.path.join(output_dir, f"{name}.csv")
        data.time_series_data.to_csv(file_path)
            
    logger.info("Data preprocessing completed")
    return train_data, val_data, test_data
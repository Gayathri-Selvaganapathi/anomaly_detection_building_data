import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Optional

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingConfig:
    dataset_name: str
    column_name: str
    model_type: str
    standardized: bool
    epochs: Union[int, None]
    batch_size: Union[int, None]
    block_size: int
    lr: Union[float, None]
    subsample: Optional[float] = 0.5
    hidden_dim: Optional[int] = 256
    n_layers: Optional[int] = 2
    max_depth: Optional[int] = 3
    n_estimators: Optional[int] = 300
    subsample: Optional[float] = 0.0
    colsample_bytree: Optional[float] = 0.0
    

class CustomDataset(Dataset):
    def __init__(self, features:np.ndarray, labels:np.ndarray):
        self.features, self.labels = features, labels

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return len(self.features)
    
class Loader:
    def __init__(self, config:TrainingConfig) -> None:
        self.config = config
        self.files = {
            "co2-1":'data/co2-1.csv',
            "co2-2":'data/co2-2.csv',
            "ele-1":'data/ele-1.csv',
            "ele-2":'data/ele-2.csv',
        }
        self.load_data(config.dataset_name, config.column_name)

    def get_total(self)->pd.DataFrame:
        df = self.raw_df[self.config.column_name].to_frame()
        self.scaler = StandardScaler()
        df["scaled_"+self.config.column_name] = self.scaler.fit_transform(df[[self.config.column_name]])
        return df

    def get_total_trainer(self):
        df = self.get_total()
        X,Y = self.batch_data_steps(df["scaled_"+self.config.column_name].values)

        return DataLoader(CustomDataset(X,Y), batch_size=self.config.batch_size, shuffle=True)

    def get_data_as_date_df(self, file_path:str) -> pd.DataFrame:
        self.raw_df = pd.read_csv(file_path)
        self.raw_df["time"] = pd.to_datetime(self.raw_df["time"])  # Convert "time" column to datetime
        self.raw_df.set_index("time", inplace=True)  # Set "time" as the index
        return self.raw_df

    def batch_data_steps(self, input_array:np.ndarray) -> tuple:
        X = [input_array[i: i+self.config.block_size] for i in range(len(input_array)-self.config.block_size)]
        Y = [input_array[i+self.config.block_size] for i in range(len(input_array)-self.config.block_size)]
        return X, Y

    def split_data(self, X:list, Y:list) -> tuple:
        # Sets ratio split
        test_ratio = int(0.9 * len(X))
        # Test train split
        X_train, X_test = X[:test_ratio], X[test_ratio:]
        y_train, y_test = Y[:test_ratio], Y[test_ratio:]

        return X_train, X_test, y_train, y_test
    
    def define_train_test_data(self) -> None:
        self.train_dataloader = DataLoader(CustomDataset(self.X_train, self.y_train), batch_size=self.config.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(CustomDataset(self.X_test, self.y_test), batch_size=self.config.batch_size, shuffle=True)

    def load_data(self, dataset_name:str, column_name:str) -> None:
        df = pd.DataFrame()
        if dataset_name in self.files.keys():
            self.dataset_name = dataset_name,
            df = self.get_data_as_date_df(self.files[dataset_name])
        else:
            raise Exception(f'File does not exist: {dataset_name}')
        if column_name not in df.columns:
            raise Exception(f'Column name does not exist in {dataset_name}: {column_name}')
    
        self.index = df.index
        arr = df[column_name].values

        if self.config.standardized:
            self.scaler = StandardScaler()
            arr = self.scaler.fit_transform(arr.reshape(-1,1)) # type: ignore

        X, Y = self.batch_data_steps(np.array(arr))
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X,Y)

        self.define_train_test_data()
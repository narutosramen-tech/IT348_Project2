import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from dataset import Dataset

class DataPreprocessor:
    """
    Handles feature scaling and dataset splitting.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def get_full_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Combine all samples into one dataset.
        Returns empty DataFrames/Series if no data is available.
        """

        X_list = []
        y_list = []

        for sample in self.dataset.samples:
            if sample.labels is not None:
                X_list.append(sample.features)
                y_list.append(sample.labels)

        # Handle empty dataset case
        if not X_list or not y_list:
            import warnings
            warnings.warn("No labeled samples found in dataset. Returning empty DataFrames.")
            # Return empty DataFrames/Series with appropriate types
            return pd.DataFrame(), pd.Series(dtype='float64')

        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)

        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, method:str = "standard"):
        """
        Scale features using training data statistics.
        Returns DataFrames with preserved column names.
        """

        if method == "standard":
            scaler = StandardScaler()

        elif method == "minmax":
            scaler = MinMaxScaler()

        else:
            raise ValueError("Unknown scaling method")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrames with original column names
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train_scaled_df, X_test_scaled_df, scaler
    
    def random_split(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Validate test_size parameter
        if not isinstance(test_size, (int, float)):
            raise TypeError("test_size must be a number")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be greater than 0 and less than 1")

        X, y = self.get_full_dataset()

        # Check if there's enough data for the specified test_size
        if len(X) < 2:
            raise ValueError(f"Not enough data for splitting. Dataset has only {len(X)} samples.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
        except ValueError as e:
            # Handle the case where stratification fails (e.g., not enough samples in test set)
            # or when stratification fails due to class imbalance
            import warnings
            warnings.warn(f"Stratification failed: {e}. Removing stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=None
            )

        return X_train, X_test, y_train, y_test
    
    def temporal_split(self, train_years, test_years) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Train on earlier years, test on later years.
        """

        # Validate input years
        if not train_years or not test_years:
            raise ValueError("train_years and test_years cannot be empty")

        if set(train_years).intersection(set(test_years)):
            raise ValueError("train_years and test_years cannot overlap")

        train_X: list[pd.DataFrame] = []
        train_y: list[pd.Series] = []
        test_X: list[pd.DataFrame] = []
        test_y: list[pd.Series] = []

        for sample in self.dataset.samples:
            if sample.year in train_years:
                train_X.append(sample.features)
                if sample.labels is not None:
                    train_y.append(sample.labels)

            elif sample.year in test_years:
                test_X.append(sample.features)
                if sample.labels is not None:
                    test_y.append(sample.labels)

        # Handle empty splits
        import warnings
        if not train_X:
            warnings.warn(f"No samples found for train_years: {train_years}")
            X_train = pd.DataFrame()
            y_train = pd.Series(dtype='float64')
        else:
            X_train = pd.concat(train_X, ignore_index=True)
            y_train = pd.concat(train_y, ignore_index=True) if train_y else pd.Series(dtype='float64')

        if not test_X:
            warnings.warn(f"No samples found for test_years: {test_years}")
            X_test = pd.DataFrame()
            y_test = pd.Series(dtype='float64')
        else:
            X_test = pd.concat(test_X, ignore_index=True)
            y_test = pd.concat(test_y, ignore_index=True) if test_y else pd.Series(dtype='float64')

        return X_train, X_test, y_train, y_test
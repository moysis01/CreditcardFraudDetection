import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import (
    SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC, SMOTEN
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks,NearMiss,

)
from imblearn.combine import SMOTEENN,SMOTETomek
from utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


def load_data(file_path):
    logger.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    #df = df.sample(frac=0.10, random_state=25)   
    logger.info("Data loaded. Shape: %s", df.shape)
    return df


# Dictionary of sampling methods
sampler_classes = {
    # Oversampling
    "SMOTE": SMOTE,
    "RANDOMOVERSAMPLER": RandomOverSampler,
    "ADASYN": ADASYN,
    "BORDERLINESMOTE": BorderlineSMOTE,
    "SVMSMOTE": SVMSMOTE,
    "KMEANSSMOTE": KMeansSMOTE,
    "SMOTENC": SMOTENC,  
    "SMOTEN": SMOTEN,       

    # Undersampling
    "RANDOMUNDERSAMPLER": RandomUnderSampler,
    "TOMEKLINKS": TomekLinks,
    "NEARMISS": NearMiss,

    # Combination
    "SMOTEENN": SMOTEENN,
    "SMOTETOMEK": SMOTETomek
}

all_scalers = {
    "MinMaxScaler": MinMaxScaler(feature_range=(-1, 1)),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}

def preprocess_data(df, config, random_state=25):
    try:
        logger.info("Starting data preprocessing...")

        original_shape = df.shape
        df.drop_duplicates(inplace=True)
        df.drop(columns='Time',inplace=True)
        logger.info(f"Dropped duplicates. Original shape: {original_shape}, New shape: {df.shape}")

        X = df.drop(['Class'], axis=1)
        y = df['Class']
        logger.info(f"Separated features and target. X shape: {X.shape}, y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
        logger.info(f"Data split into train and test sets. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        logger.info(f"Class distribution before resampling: {y_train.value_counts().to_dict()}")
        # Resampling (if specified in the config)
        sampling_methods = config.get('resampling', [])
        resampling_params = config.get('resampling_params', {})

        for method in sampling_methods:
            logger.info(f"Applying resampling method: {method}")
            sampler_class = sampler_classes.get(method.upper())
            if sampler_class:
                # Initialize the sampler with parameters if provided
                method_params = resampling_params.get(method.upper(), {})
                sampler = sampler_class(**method_params)
                start_time = time.time()
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                end_time = time.time()
                logger.info(f"Resamlping time with {method}: {end_time - start_time:.2f} seconds")

                logger.info(f"Class distribution after resampling: {y_train.value_counts().to_dict()}")
            else:
                logger.warning(f"Invalid resampling method '{method}' specified in config. Skipping resampling.")


        # Scaling
        scaler_name = config.get('scaling', None)
        if scaler_name:
            logger.info("Scaling features...")
            scaler_class = all_scalers.get(scaler_name)
            if not scaler_class:
                raise ValueError(f"Invalid scaler '{scaler_name}' in config.")
            scaler = scaler_class.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logger.info(f"Scaled features with {scaler_name}. Shapes - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")

            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        else:
            logger.info("No scaling applied. Proceeding without scaling.")
            X_train_scaled = X_train
            X_test_scaled = X_test

        logger.info("Preprocessing completed successfully.")
        return X_train_scaled, X_test_scaled, y_train, y_test, X, y

    except Exception as e:
        logger.error("An error occurred during preprocessing: %s", e)
        raise
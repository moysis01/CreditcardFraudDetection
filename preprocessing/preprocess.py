import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import (
    SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

def load_data(file_path):
    logger.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    df = df.sample(frac=0.50, random_state=25)
    logger.info("Data loaded. Shape: %s", df.shape)
    return df

# Dictionary of sampling methods
sampler_classes = {
    # Oversampling
    "SMOTE": SMOTE,
    "ROS": RandomOverSampler,
    "ADASYN": ADASYN,
    "BLSMOTE": BorderlineSMOTE,
    "SVMSMOTE": SVMSMOTE,
    "KMEANSSMOTE": KMeansSMOTE,

    # Undersampling
    "RUS": RandomUnderSampler,
    "NEARMISS": NearMiss,

    # Combination of both
    "SMOTEENN": SMOTEENN,
    "SMOTETOMEK": SMOTETomek
}


def preprocess_data(df, config, random_state=25):
    try:
        logger.info("Starting data preprocessing...")

        original_shape = df.shape
        df.drop_duplicates(inplace=True)
        df.drop(columns='Time', inplace=True)
        logger.info(f"Dropped duplicates. Original shape: {original_shape}, New shape: {df.shape}")

        X = df.drop(['Class'], axis=1)
        y = df['Class']
        logger.info(f"Separated features and target. X shape: {X.shape}, y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
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
                logger.info(f"Resampling time with {method}: {end_time - start_time:.2f} seconds")

                logger.info(f"Class distribution after resampling: {y_train.value_counts().to_dict()}")
            else:
                logger.warning(f"Invalid resampling method '{method}' specified in config. Skipping resampling.")

        # Scaling
        if config.get('scaling', False):
            logger.info("Scaling features...")
            standard_scaler = StandardScaler()
            robust_scaler = RobustScaler()
            minmax_scaler= MinMaxScaler(feature_range=(-1,1))

            # Scale V1 to V28 with StandardScaler
            features_v1_v28 = [f'V{i}' for i in range(1, 29)]
            X_train_v1_v28 = standard_scaler.fit_transform(X_train[features_v1_v28])
            X_test_v1_v28 = standard_scaler.transform(X_test[features_v1_v28])

            # Scale Amount with RobustScaler
            X_train_amount = robust_scaler.fit_transform(X_train[['Amount']])
            X_test_amount = robust_scaler.transform(X_test[['Amount']])

            # Combine scaled features
            X_train_scaled = pd.DataFrame(X_train_v1_v28, columns=features_v1_v28)
            X_train_scaled['Amount'] = X_train_amount
            X_test_scaled = pd.DataFrame(X_test_v1_v28, columns=features_v1_v28)
            X_test_scaled['Amount'] = X_test_amount

            logger.info(f"Scaled features. Shapes - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
        else:
            logger.info("No scaling applied. Proceeding without scaling.")
            X_train_scaled = X_train
            X_test_scaled = X_test

        logger.info("Preprocessing completed successfully.")
        return X_train_scaled, X_test_scaled, y_train, y_test, X, y

    except Exception as e:
        logger.error("An error occurred during preprocessing: %s", e)
        raise

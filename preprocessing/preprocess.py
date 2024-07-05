import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
from utils.progress_logger import ProgressLogger

logger = setup_logger(__name__)

def load_data(file_path):
    logger.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    #df = df.sample(frac=0.10, random_state=25)
    logger.info("Data loaded. Shape: %s", df.shape)
    return df

def preprocess_data(df):
    try:
        total_steps = 7
        progress_logger = ProgressLogger(logger, total_steps)

        progress_logger.log_step("Starting data preprocessing...")

        original_shape = df.shape
        df.drop_duplicates(inplace=True)
        progress_logger.log_step(f"Dropped duplicates. Original shape: {original_shape}, New shape: {df.shape}")

        X = df.drop('Class', axis=1)
        y = df['Class']
        progress_logger.log_step(f"Separated features and target. X shape: {X.shape}, y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
        progress_logger.log_step(f"Data split into train and test sets. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        progress_logger.log_step("Applying SMOTETomek...")
        preprocessor = SMOTETomek(random_state=25)
        X_train_res, y_train_res = preprocessor.fit_resample(X_train, y_train)
        progress_logger.log_step(f"Applied SMOTETomek. Resampled X_train shape: {X_train_res.shape}, y_train shape: {y_train_res.shape}")

        progress_logger.log_step("Scaling features...")
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_res)
        X_train_scaled = scaler.transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        progress_logger.log_step(f"Scaled features. Shapes - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        progress_logger.log_step("Preprocessing completed successfully.")

        return X_train_scaled, X_test_scaled, y_train_res, y_test, X, y
    except Exception as e:
        logger.error("An error occurred during preprocessing: %s", e)
        raise

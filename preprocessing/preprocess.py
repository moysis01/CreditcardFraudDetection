import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_data(file_path):
    logger.info("Loading data from file: %s", file_path)
    # Corrected comment for accurate sampling fraction description
    df = pd.read_csv(file_path)
    #df = df.sample(frac=0.30, random_state=42)  # Load only 10% of the data for testing
    logger.info("Data loaded and sampled. Shape: %s", df.shape)
    return df

def preprocess_data(df):
    try:
        logger.info("Starting data preprocessing...")
        df = df.drop_duplicates()
        logger.info("Dropped duplicates. New DataFrame Shape: %s", df.shape)

        X = df.drop('Class', axis=1)
        y = df['Class']
        logger.info("Separated features and target. X shape: %s, y shape: %s", X.shape, y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        logger.info("Data split into train and test sets. X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)

        preprocessor = SMOTETomek(random_state=1)
        X_train_res, y_train_res = preprocessor.fit_resample(X_train, y_train)
        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
        y_train_res = pd.Series(y_train_res)

        logger.info("Applied SMOTETomek. Resampled X_train shape: %s, y_train shape: %s", X_train_res.shape, y_train_res.shape)

        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)  # Ensure scaler is fitted only on training data
        X_train_scaled = scaler.transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back to DataFrame for compatibility with many sklearn functions
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        logger.info("Scaled features. Shapes - X_train: %s, X_test: %s", X_train_scaled.shape, X_test_scaled.shape)
        return X_train_scaled, X_test_scaled, y_train_res, y_test
    except Exception as e:
        logger.error("An error occurred during preprocessing: %s", e)
        raise

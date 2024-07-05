import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)

def load_data(file_path):
    logger.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    df = df.sample(frac=0.3, random_state=25)  # Use 30% of the data to avoid SMOTETomek issues
    logger.info("Data loaded. Shape: %s", df.shape)
    return df

def preprocess_data(df):
    try:
        logger.info("Starting data preprocessing...")

        original_shape = df.shape
        df.drop_duplicates(inplace=True)
        logger.info(f"Dropped duplicates. Original shape: {original_shape}, New shape: {df.shape}")

        X = df.drop('Class', axis=1)
        y = df['Class']
        logger.info(f"Separated features and target. X shape: {X.shape}, y shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
        logger.info(f"Data split into train and test sets. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        logger.info("Applying SMOTETomek...")
        preprocessor = SMOTETomek(random_state=25)
        total_chunks = 10
        chunk_size = len(X_train) // total_chunks
        X_train_res = pd.DataFrame()
        y_train_res = pd.Series(dtype='int')

        for i in tqdm(range(total_chunks), desc="Applying SMOTETomek", ncols=100):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size if i < total_chunks - 1 else len(X_train)

            X_chunk, y_chunk = preprocessor.fit_resample(X_train.iloc[start_index:end_index], y_train.iloc[start_index:end_index])
            X_train_res = pd.concat([X_train_res, X_chunk])
            y_train_res = pd.concat([y_train_res, y_chunk])

        logger.info(f"Applied SMOTETomek. Resampled X_train shape: {X_train_res.shape}, y_train shape: {y_train_res.shape}")

        logger.info("Scaling features...")
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_res)
        X_train_scaled = scaler.transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        logger.info(f"Scaled features. Shapes - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        logger.info("Preprocessing completed successfully.")

        return X_train_scaled, X_test_scaled, y_train_res, y_test, X, y
    except Exception as e:
        logger.error("An error occurred during preprocessing: %s", e)
        raise

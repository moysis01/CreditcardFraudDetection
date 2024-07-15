import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_data(file_path):
    logger.info("Loading data from file: %s", file_path)
    df = pd.read_csv(file_path)
    #df = df.sample(frac=0.50, random_state=25)  # Use 50% of the data 
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)
        logger.info(f"Data split into train and test sets. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        logger.info("Class distribution before SMOTE: %s", y_train.value_counts().to_dict())

        logger.info("Applying SMOTE...")
        smote = SMOTE(random_state=25)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"Applied SMOTE. Resampled X_train shape: {X_train_res.shape}, y_train shape: {y_train_res.shape}")

        logger.info("Class distribution after SMOTE: %s", y_train_res.value_counts().to_dict())

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

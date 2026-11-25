
import pandas as pd
from pathlib import Path


def preprocess_data():

    # Define project directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    # Load datasets
    train = pd.read_csv(RAW_DIR / "train.csv")
    meal = pd.read_csv(RAW_DIR / "meal_info.csv")
    center = pd.read_csv(RAW_DIR / "fulfilment_center_info.csv")

    # Merge training data with meal information
    df = pd.merge(train, meal, on="meal_id", how="left")

    # Merge fulfilment centre information
    df = pd.merge(df, center, on="center_id", how="left")

    # Handle missing values
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicate records
    df = df.drop_duplicates()

    # Create processed directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(PROCESSED_DIR / "processed_data.csv", index=False)

    print("Data preprocessing completed successfully!")
    print("Final dataset shape:", df.shape)

    return df


if __name__ == "__main__":
    preprocess_data()

import pandas as pd
import os


def preprocess_data(input_path: str, output_path: str) -> None:
    """
    Fungsi untuk melakukan preprocessing dataset Titanic
    """

    # Load data
    df = pd.read_csv(input_path)

    # Drop kolom yang tidak relevan
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encoding categorical features
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked"],
        drop_first=True
    )

    # Simpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Preprocessing selesai.")
    print(f"Dataset disimpan di: {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    INPUT_PATH = os.path.join(
        BASE_DIR,
        "..",
        "kaggle_raw",
        "train.csv"
    )

    OUTPUT_PATH = os.path.join(
        BASE_DIR,
        "..",
        "kaggle_preprocessing",
        "titanic_preprocessed.csv"
    )

    preprocess_data(INPUT_PATH, OUTPUT_PATH)







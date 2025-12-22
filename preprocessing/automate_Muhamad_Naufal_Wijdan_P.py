import pandas as pd
import os

def preprocess_data(input_path: str, output_path: str) -> None:
    print(f"Membaca data dari: {input_path}")
    df = pd.read_csv(input_path)

    # Preprocessing Sederhana
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, errors='ignore')
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Simpan Hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Berhasil! File disimpan di: {output_path}")

if __name__ == "__main__":
    # Lokasi folder preprocessing
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Pastikan nama file ini sesuai dengan yang ada di folder GitHub kamu!
    INPUT_PATH = os.path.join(BASE_DIR, "train.csv") 
    
    # 2. Path output sesuai keinginanmu
    OUTPUT_PATH = os.path.join(BASE_DIR, "titanic_preprocessing.csv")

    if os.path.exists(INPUT_PATH):
        preprocess_data(INPUT_PATH, OUTPUT_PATH)
    else:
        print(f"ERROR: File '{INPUT_PATH}' tidak ditemukan!")
        print(f"Isi folder saat ini: {os.listdir(BASE_DIR)}")

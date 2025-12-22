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
  
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    ROOT_DIR = os.path.dirname(BASE_DIR)

    INPUT_PATH = os.path.join(ROOT_DIR, "train.csv") 
    
    OUTPUT_PATH = os.path.join(BASE_DIR, "titanic_preprocessing.csv")

    print(f"DEBUG: Mencari dataset di: {INPUT_PATH}")

    if os.path.exists(INPUT_PATH):
        preprocess_data(INPUT_PATH, OUTPUT_PATH)
    else:
        print(f"ERROR: File '{INPUT_PATH}' tidak ditemukan!")
       
        print(f"Isi folder ROOT: {os.listdir(ROOT_DIR)}")

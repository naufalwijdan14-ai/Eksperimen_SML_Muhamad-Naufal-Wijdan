import pandas as pd
import os

def preprocess_data(input_path: str, output_path: str) -> None:
    """
    Fungsi untuk melakukan preprocessing dataset Titanic
    """
    print(f"Membaca data dari: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)

    # Drop kolom yang tidak relevan
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, errors='ignore')

    #  Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # 3. Encoding categorical features 
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked"],
        drop_first=True
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("--- Preprocessing Selesai ---")
    print(f"Dataset berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    

    ROOT_DIR = os.path.dirname(BASE_DIR)

    INPUT_PATH = os.path.join(ROOT_DIR, "kaggle_raw", "train.csv")
    OUTPUT_PATH = os.path.join(ROOT_DIR, "preprocessing", "titanic_preprocessed.csv")

    # Debugging Info (Akan muncul di Log GitHub Actions)
    print(f"DEBUG: Root Directory -> {ROOT_DIR}")
    print(f"DEBUG: Input Path -> {INPUT_PATH}")

    # Validasi keberadaan file sebelum diproses
    if os.path.exists(INPUT_PATH):
        preprocess_data(INPUT_PATH, OUTPUT_PATH)
    else:
        print(f"ERROR: File '{INPUT_PATH}' tidak ditemukan!")
        print("Pastikan struktur folder di GitHub adalah: kaggle_raw/train.csv")
        
        if os.path.exists(ROOT_DIR):
            print(f"Isi folder ROOT saat ini: {os.listdir(ROOT_DIR)}")

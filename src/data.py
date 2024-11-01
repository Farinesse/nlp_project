import pandas as pd
from sklearn.model_selection import train_test_split

"""
def make_dataset(filename):
    df = pd.read_csv(filename)
    X, y = df["video_name"].values, df["is_comic"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df_train = pd.DataFrame({"video_name": X_train, "label": y_train})
    df_test = pd.DataFrame({"video_name": X_test, "label": y_test})

    df_train.to_csv("src/data/raw/train.csv", index=False)
    df_test.to_csv("src/data/raw/test.csv", index=False)

    return df_train, df_test"""

def make_dataset(filename,split: bool = False):

    try :
        df = pd.read_csv(filename)
        if "video_name" not in df.columns or "is_comic" not in df.columns:
            raise ValueError("Le fichier CSV doit contenir les colonnes 'video_name' et 'is_comic'.")

        df["video_name"] = df["video_name"].fillna("")
        df["is_comic"] = pd.to_numeric(df["is_comic"], errors="coerce").fillna(0).astype(int)

        if split:
            # Split the data
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["is_comic"]
            )
            train_df.to_csv("data/raw/train.csv", index=False)
            test_df.to_csv("data/raw/test.csv", index=False)
            return train_df

        return df

    except Exception as e :
        raise ValueError("Le fichier CSV est mal structuré")

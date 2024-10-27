import os

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score




from data import make_dataset

from feature import (
    make_features,
    process_text_lematization,
)

from models import make_model


VALID_ENCODERS = ["word2vec", "count", "tfidf"]
VALID_MODELS = ["svc", "random_forest", "logistic_regression"]

DEFAULT_PATHS = {
    "model": "models/dump.json",
    "vectorizer": {
        "word2vec": "models/word2vec.model",
        "count": "models/count_vectorizer.pkl",
        "tfidf": "models/tfidf_vectorizer.pkl"
    }
}

def ensure_dir(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--input_filename", default="data/raw/train.csv", help="File training data"
)
@click.option(
    "--model_dump_filename",
    default="models/dump.json",
    help="File to dump model",
)
@click.option(
    "--encoder",
    type=click.Choice(VALID_ENCODERS),
    default="count",
    help="Type of text encoder to use"
)
@click.option(
    "--model_type",
    type=click.Choice(VALID_MODELS),
    default="random_forest",
    help="Type of model to train"
)
def train(input_filename: str, model_dump_filename: str, encoder: str, model_type: str ) -> None:

    ensure_dir(model_dump_filename)
    df = make_dataset(input_filename)
    vectorizer_path = DEFAULT_PATHS["vectorizer"][encoder]
    X_train, y_train = make_features(df, save_path=vectorizer_path, vectorizer_type=encoder)
    model = make_model(model_type)
    model.fit(X_train, y_train)
    save_dict = {
        "model": model,
        "encoder_type": encoder,
        "vectorizer_path": vectorizer_path
    }
    print(f"Saving model and configuration to {model_dump_filename}")
    joblib.dump(save_dict, model_dump_filename)
    print("Training completed successfully!")




@click.command()
@click.option(
    "--input_filename",
    default="data/raw/test.csv",
    help="File training data"
)
@click.option(
    "--model_dump_filename",
    default="models/dump.json",
    help="File to dump model"
)
@click.option(
    "--output_filename",
    default="data/processed/prediction.csv",
    help="Output file for predictions",
)
def predict(input_filename, model_dump_filename, output_filename):
    try:
        ensure_dir(output_filename)

        # Charger le modèle et sa configuration
        print(f"Loading model from {model_dump_filename}")
        if not os.path.exists(model_dump_filename):
            raise FileNotFoundError(f"Model file not found: {model_dump_filename}")

        saved_data = joblib.load(model_dump_filename)

        # Extraire le modèle et l'encoder utilisé lors de l'entraînement
        if isinstance(saved_data, dict) and "model" in saved_data:
            print("Loading model with configuration...")
            model = saved_data["model"]
            encoder = saved_data.get("encoder_type")

            # Reconstruire le chemin du vectorizer à partir de l'encodeur
            vectorizer_path = DEFAULT_PATHS["vectorizer"][encoder]

            if not encoder:
                raise ValueError("Model file is missing encoder configuration")

            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

            print(f"Using vectorizer: {vectorizer_path}")
        else:
            raise ValueError("Invalid model format. Please retrain the model.")

        print(f"Loading test data from {input_filename}")
        df_test = pd.read_csv(input_filename)

        print(f"Creating features using {encoder} encoder...")
        df_test_features, y = make_features(
            df_test,
            train=False,
            save_path=vectorizer_path,
            vectorizer_type=encoder
        )

        print("Making predictions...")
        predictions = model.predict(df_test_features)

        results = pd.DataFrame({
            "id": df_test.index.copy(),
            "prediction": predictions
        })

        if "is_comic" in df_test.columns:
            accuracy = accuracy_score(df_test["is_comic"], predictions)
            print(f"\nAccuracy test: {accuracy:.4f}")

        print(f"Saving predictions to {output_filename}")
        results.to_csv(output_filename, index=False)
        print("Predictions completed successfully!")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
@click.command()
@click.option(
    "--input_filename", default="data/raw/train.csv", help="File training data"
)
@click.option(
    "--encoder",
    type=click.Choice(VALID_ENCODERS),
    default="count",
    help="Type of text encoder to use"
)
@click.option(
    "--model_type",
    type=click.Choice(VALID_MODELS),
    default="random_forest",
    help="Type of model to evaluate"
)

def evaluate(input_filename, encoder: str = "count",model_type: str = "random_forest"):
    try:
        df_train = make_dataset(input_filename)
        vectorizer_path = DEFAULT_PATHS["vectorizer"][encoder]

        X_train, y_train = make_features(df_train,save_path=vectorizer_path,)
        print("evalute : ", model_type )
        model = make_model(model_type)
        model.fit(X_train, y_train)
        return evaluate_model(model, X_train, y_train)
    except Exception as e :
        print(f"Error during evaluation: {str(e)}")
        raise





def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Scores pour chaque pli (fold): {scores}")
    print(f"Score moyen : {np.mean(scores)}")
    print(f"Standard deviation: {np.std(scores):.4f}")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
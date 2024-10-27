# NLP Video Classification Project

Ce projet implémente une pipeline de classification de vidéos basée sur le traitement du langage naturel (NLP). Il permet de classifier des vidéos comme étant comiques ou non en se basant sur leurs titres.

## Structure du Projet

```
NLP_project/
├── data/
│   ├── processed/       # Données traitées et prédictions
│   └── raw/            # Données brutes (CSV)
│       ├── train.csv   # Fichier d'entraînement
│       └── test.csv    # Fichier de test
├── models/             # Modèles entraînés et vectorizers
├── notebook/          # Notebooks Jupyter pour l'exploration
├── src/               # Code source
│   ├── data.py        # Fonctions de chargement des données
│   ├── feature.py     # Fonctions de feature engineering
│   ├── models.py      # Définitions des modèles
│   └── main.py        # Point d'entrée principal
├── .gitignore
├── README.md
└── requirements.txt
```

## Prérequis

### Données Requises

Avant de commencer, assurez-vous d'avoir les fichiers de données suivants dans le dossier `data/raw/` :

1. **train.csv** : Fichier d'entraînement avec les colonnes :
   - `video_name` : Titre de la vidéo
   - `is_comic` : Label (1 pour comique, 0 pour non-comique)

2. **test.csv** : Fichier de test avec la colonne :
   - `video_name` : Titre de la vidéo
   - `is_comic` (optionnel) : Pour l'évaluation

Format des fichiers CSV :
```csv
video_name,is_comic
"Titre de la vidéo 1",1
"Titre de la vidéo 2",0
```

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/Farinesse/nlp_project.git
cd NLP_project
```

2. Créer l'environnement virtuel et l'activer :
```bash
python -m venv venv

# Sur Windows :
venv\Scripts\activate
# Sur Unix/MacOS :
source venv/bin/activate
```

3. Mettre à jour pip :
```bash
python -m pip install --upgrade pip
```

4. Installer les dépendances :
```bash
pip install -r requirements.txt
```

5. Télécharger le modèle spaCy français :
```bash
python -m spacy download fr_core_news_md
```

## Utilisation

### Entraînement d'un Modèle

```bash
python src/main.py train --encoder [ENCODER] --model_type [MODEL]
```

Options disponibles :
- `--encoder` : Type d'encodeur de texte ('word2vec', 'count', 'tfidf')
- `--model_type` : Type de modèle ('random_forest', 'svc', 'logistic_regression')
- `--input_filename` : Chemin vers les données d'entraînement
- `--model_dump_filename` : Chemin pour sauvegarder le modèle

Exemple :
```bash
python src/main.py train --encoder word2vec --model_type random_forest
```

### Faire des Prédictions

```bash
python src/main.py predict [OPTIONS]
```

Options :
- `--input_filename` : Fichier de données de test
- `--model_dump_filename` : Fichier du modèle entraîné
- `--output_filename` : Fichier de sortie pour les prédictions

### Évaluation du Modèle

```bash
python src/main.py evaluate [OPTIONS]
```

Options :
- `--encoder` : Type d'encodeur à évaluer
- `--model_type` : Type de modèle à évaluer
- `--input_filename` : Données pour l'évaluation

## Encodeurs Disponibles

1. **Count Vectorizer** (`count`)
   - Vectorisation basée sur la fréquence des mots
   - Bon pour les textes courts

2. **TF-IDF** (`tfidf`)
   - Vectorisation tenant compte de l'importance relative des mots
   - Performant pour les textes de longueur variable

3. **Word2Vec** (`word2vec`)
   - Embeddings de mots basés sur le contexte
   - Capture mieux les relations sémantiques

## Modèles Disponibles

1. **Random Forest** (`random_forest`)
   - Bon équilibre entre performance et interprétabilité
   - Gère bien les features non linéaires

2. **SVC** (`svc`)
   - Support Vector Classification
   - Performant sur les espaces de grande dimension

3. **Logistic Regression** (`logistic_regression`)
   - Modèle linéaire simple et interprétable
   - Bon pour les données linéairement séparables



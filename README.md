# University Assistant RAG – MLOps

Assistant conversationnel qui répond aux questions universitaires à partir de documents institutionnels publics, avec citations et refus lorsque les sources sont insuffisantes.

## Architecture

- Collecte : pages HTML et PDF officiels
- Extraction : BeautifulSoup et PyMuPDF
- Prétraitement : normalisation et découpage en chunks
- Embeddings : Sentence Transformers
- Recherche vectorielle : FAISS CPU
- Génération : Ollama (`qwen2.5:3b`)
- API : FastAPI
- Interface : Streamlit
- MLOps : MLflow, DVC, pytest, GitHub Actions, logs structurés

> Le dépôt contient le code et des sources publiques configurées. Les documents ne sont pas inclus : télécharge-les avec le script. Vérifie toujours leur date et leur statut avant de les utiliser comme règles actuelles.

## Prérequis

- Ubuntu/Linux
- Python 3.10+ (Python 3.12 recommandé)
- 16 Go RAM recommandés
- Ollama installé
- Connexion Internet pour la collecte initiale

## Installation

```bash
git clone <URL_DU_DEPOT>
cd university-assistant-rag

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull qwen2.5:3b
ollama serve
```

Dans un autre terminal :

```bash
cp .env.example .env
python scripts/collect_documents.py
python scripts/build_index.py
```

## Démarrer

API :

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Interface :

```bash
streamlit run app/streamlit_app.py
```

## Évaluer

```bash
pytest -q
python scripts/evaluate.py
```

MLflow :

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

## Versionner les données avec DVC

```bash
dvc init
dvc add data/raw data/processed artifacts
git add .dvc .gitignore data/raw.dvc data/processed.dvc artifacts.dvc
git commit -m "chore: track data and RAG artifacts with DVC"
```

Adapte les chemins si tu ne souhaites versionner que certains répertoires. Ne pousse pas de données personnelles ni de documents dont la redistribution n'est pas autorisée.

## Structure

```text
university-assistant-rag/
├── app/
├── configs/
├── data/{raw,processed,evaluation}/
├── docs/
├── artifacts/
├── scripts/
├── src/{api,ingestion,rag,monitoring}/
├── tests/
├── .github/workflows/
├── requirements.txt
└── README.md
```

## Limites et précautions

- Prototype académique, pas un service officiel de l'établissement.
- Ne pas charger de dossiers étudiants, notes, identifiants ou données sensibles.
- Les documents archivés doivent être étiquetés et ne pas servir à répondre sur les procédures courantes.
- Le système doit citer les documents récupérés et signaler les dates inconnues.
- Un LLM peut produire des erreurs : ne pas utiliser ses réponses pour une décision administrative ou médicale.

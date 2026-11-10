# Architecture technique

```text
Sources officielles PDF/HTML
          |
          v
Collecte + inventaire SHA-256
          |
          v
Extraction / nettoyage / normalisation
          |
          v
Découpage en chunks avec chevauchement
          |
          v
Embeddings multilingues (CPU)
          |
          v
FAISS (un index vectoriel)
          |
Question -> embedding -> top-k -> contexte
                                  |
                                  v
                           Ollama / Qwen
                                  |
                                  v
                  Réponse + sources / refus
                                  |
                     FastAPI + Streamlit
```

## Décisions
- Un seul index FAISS pour respecter le périmètre du projet.
- Embeddings multilingues pour préparer l'extension français/arabe/darija.
- Exécution CPU pour compatibilité avec un GPU Intel intégré.
- Ollama pour servir le LLM localement.
- Métadonnées de source conservées au niveau des chunks.
- Les dates et statuts documentaires doivent être enrichis avant toute utilisation réelle.

## MLOps à compléter pendant le projet
1. DVC pour les jeux de données et artefacts.
2. MLflow pour comparer `chunk_size`, `overlap`, `top_k` et modèles.
3. Tests d'intégration de collecte, retrieval et API.
4. Quality gates : seuil Recall@k, taux de citations valides, refus hors corpus.
5. Monitoring : latence, erreurs API, taille du corpus et ressources CPU/RAM.

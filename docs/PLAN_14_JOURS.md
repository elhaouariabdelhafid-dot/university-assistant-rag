# Plan de réalisation sur 14 jours — Suivi actualisé

## Objectif

Construire et industrialiser un assistant RAG universitaire en respectant les neuf étapes méthodologiques et les livrables définis par l'enseignant.

Le tableau ci-dessous conserve le découpage initial en 14 journées de travail. Il présente l'état actuel des activités ; il ne correspond pas nécessairement à 14 jours calendaires déjà écoulés.

## Plan et état d'avancement

| Jour | Activité | État actuel |
|---|---|---|
| 1 | Cadrage métier, périmètre, exigences | Charte initiale réalisée ; formalisation en cours |
| 2 | Inventaire et provenance documentaire | Corpus identifié ; inventaire structuré à compléter |
| 3 | Collecte et préparation des documents | Corpus de test disponible |
| 4 | Extraction, nettoyage et contrôle qualité | Pipeline développé ; contrôles à documenter |
| 5 | Chunking et métadonnées | Implémentés ; reproductibilité à consolider |
| 6 | Embeddings et index FAISS | Réalisés |
| 7 | Retrieval et génération LLM | Réalisés avec Gemini et Qwen optionnel |
| 8 | Citations, refus et limites | Citations et refus implémentés ; contradictions à traiter |
| 9 | Expériences MLflow | Baseline et expansion contextuelle enregistrées |
| 10 | Évaluation retrieval et génération | Première évaluation réalisée ; mesures avancées à ajouter |
| 11 | DVC et reproductibilité | À réaliser |
| 12 | Tests automatisés, CI et quality gates | À réaliser |
| 13 | API, interface, déploiement et logs | À réaliser |
| 14 | Gouvernance, rapport et démonstration | À finaliser |

## Résultats techniques obtenus

- Corpus : huit documents fictifs.
- Index FAISS : 18 chunks.
- Embeddings multilingues.
- Pipeline Gemini avec citations.
- Expansion contextuelle.
- Évaluation fonctionnelle sur 10 questions.
- Amélioration observée : 90 % vers 100 %.
- Suivi MLflow opérationnel.
- Code versionné et publié sur GitHub.

## Priorités restantes

### Priorité 1 — Compléter la documentation

- Valider la charte métier.
- Créer l'inventaire documentaire.
- Documenter les paramètres d'ingestion.
- Mettre à jour l'architecture.
- Ajouter un fichier `.env.example` sans secrets.

### Priorité 2 — Robustesse et évaluation

- Tester le retrieval insuffisant.
- Tester les questions hors périmètre.
- Tester les contradictions.
- Évaluer Faithfulness et complétude.
- Vérifier la justification des citations.
- Formaliser les expériences comparatives.

### Priorité 3 — Industrialisation

- Ajouter DVC.
- Créer les tests Pytest.
- Définir les quality gates.
- Configurer GitHub Actions.
- Développer FastAPI et Streamlit.
- Préparer le déploiement Docker.
- Ajouter les logs et le monitoring.

### Priorité 4 — Livrables

- README complet.
- Rapport technique de 8 à 12 pages.
- Dossier de reproductibilité.
- Résultats et graphiques d'évaluation.
- Rapport de risques et gouvernance.
- Vidéo de démonstration de 2 minutes.

## Critères de fin de projet

Le projet sera considéré comme livrable lorsque :

1. Le corpus, le code et les configurations seront reproductibles.
2. Les tests de retrieval, de génération et de robustesse seront documentés.
3. Les quality gates seront automatisés dans la CI.
4. L'application sera exécutable selon les instructions du README.
5. Les limites et les risques seront explicitement présentés.
6. Tous les livrables demandés par l'enseignant seront disponibles.

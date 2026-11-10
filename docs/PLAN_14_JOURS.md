# Plan de réalisation sur 14 jours

| Jour | Travail | Livrable |
|---|---|---|
| 1 | Cadrage, exigences, risques | Charte projet |
| 2 | Sources officielles et inventaire | `documents.csv` |
| 3 | Collecte PDF/HTML | Corpus brut |
| 4 | Extraction et exploration | Textes extraits + rapport |
| 5 | Nettoyage et chunking | Chunks reproductibles |
| 6 | Embeddings et index FAISS | Index local |
| 7 | Retrieval + Ollama | RAG fonctionnel |
| 8 | Citations et refus | Réponses traçables |
| 9 | MLflow et expériences | Comparaison de configurations |
| 10 | Jeu d'évaluation | Recall@k et analyse qualitative |
| 11 | DVC et reproductibilité | Données versionnées |
| 12 | Tests et CI | Pipeline de tests |
| 13 | API, interface et logs | Démo utilisable |
| 14 | Rapport, gouvernance et vidéo | Rapport 8–12 pages + démo |

## Critères d'acceptation
- Chaque réponse contient des sources identifiables.
- Le système refuse les questions hors corpus.
- Les documents archivés sont signalés.
- L'index peut être reconstruit depuis les données.
- Les tests passent en CI.
- Les résultats d'évaluation et les limites sont documentés.

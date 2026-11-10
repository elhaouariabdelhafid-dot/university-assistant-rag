# Suivi MLflow – protocole proposé

Expériences à comparer :
- chunk_size : 120, 180, 250 mots
- overlap : 20, 35, 50 mots
- top_k : 3, 5, 8
- embedding : modèle multilingue MiniLM puis modèle alternatif si les ressources le permettent

À logger :
- paramètres de chunking et modèle
- nombre de documents/chunks
- Recall@k sur questions annotées
- latence moyenne et p95
- taux de refus hors corpus
- précision des citations évaluée manuellement

Ne conclure qu'à partir d'un jeu de test stable, séparé des questions utilisées pour régler les paramètres.

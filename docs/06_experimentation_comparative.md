
# Étape 6 — Expérimentation comparative du système RAG

## 1. Objectif

L'objectif est de comparer deux configurations de découpage
documentaire afin d'étudier leur influence sur les performances
de recherche du système University Assistant RAG.

L'expérimentation utilise le même corpus, le même modèle
d'embeddings, le même index vectoriel et le même jeu de
questions pour les deux configurations.

Les résultats sont suivis avec MLflow.

## 2. Corpus expérimental

- Université fictive : Université Al Amal
- Documents : 8
- Nombre de mots source : 2 169
- Format : TXT
- Langue principale : français
- Origine : documents synthétiques de test

Le corpus ne contient pas de données universitaires officielles.

## 3. Configurations comparées

| Paramètre | A_reference | B_experimentale |
|---|---|---|
| Chunk size | 180 mots | 120 mots |
| Chunk overlap | 35 mots | 20 mots |
| Chunks obtenus | 18 | 24 |
| Embeddings | MiniLM multilingue | MiniLM multilingue |
| Index vectoriel | FAISS IndexFlatIP | FAISS IndexFlatIP |
| Normalisation | Oui | Oui |
| Top K | 5 | 5 |
| Questions | 8 | 8 |

Modèle d'embeddings :

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Les index expérimentaux sont construits en mémoire.
L'index FAISS de référence utilisé par l'application
n'est pas modifié.

## 4. Méthodologie

Pour chaque configuration :

1. Charger les huit documents TXT.
2. Découper les textes selon les paramètres choisis.
3. Calculer les embeddings normalisés des chunks.
4. Construire un index FAISS IndexFlatIP.
5. Réutiliser les mêmes embeddings de questions.
6. Effectuer une recherche Top 5.
7. Comparer les documents récupérés avec les références attendues.
8. Calculer les métriques de qualité et de latence.
9. Enregistrer les paramètres et résultats dans MLflow.

Le jeu de huit questions couvre les huit documents du corpus.

Les métriques sont calculées au niveau des documents :
un résultat est considéré pertinent si son document source
correspond au document attendu.

La pertinence exacte du chunk et la fidélité des réponses
générées ne sont pas évaluées ici.

## 5. Métriques

### Hit@1

Proportion de questions pour lesquelles le document attendu
apparaît en première position.

### Recall@5 documentaire

Proportion de questions pour lesquelles le document attendu
apparaît parmi les cinq premiers chunks récupérés.

Cette définition est opérationnelle pour notre jeu à un
document attendu par question.

### MRR@5

Moyenne des inverses des rangs du premier chunk appartenant
au document attendu, dans les cinq premiers résultats.

### Latence FAISS

Temps moyen de recherche vectorielle dans l'index,
hors calcul des embeddings.

Cette mesure n'est pas équivalente à la latence
de bout en bout de l'assistant.

## 6. Résultats expérimentaux MLflow

| Métrique | Configuration A | Configuration B |
|---|---:|---:|
| Nombre de chunks | 18 | 24 |
| Hit@1 | 0.7500 | 0.6250 |
| Recall@5 | 1.0000 | 1.0000 |
| MRR@5 | 0.8750 | 0.7917 |
| Latence FAISS (ms) | 0.0270 | 0.0489 |

Les valeurs correspondent aux exécutions enregistrées
dans MLflow le 10 novembre 2026.

## 7. Analyse des résultats

### 7.1 Effet du chunking

La configuration B produit 24 chunks, contre 18 pour A.

Le découpage plus fin n'améliore pas le classement
sur les huit questions testées.

### 7.2 Qualité de recherche

La configuration A atteint un Hit@1 de 75 %,
contre 62,5 % pour B.

Cela représente six questions avec le document attendu
au premier rang pour A, contre cinq pour B.

Le MRR@5 est également plus élevé pour A :
0.8750 contre 0.7917.

Les deux configurations obtiennent 100 % au Recall@5
documentaire sur le jeu utilisé.

### 7.3 Exemple d'erreur

Question :

Comment obtenir une attestation de scolarité ?

Configuration A :
le document DOC08_frais_procedures apparaît au rang 1.

Configuration B :
le document DOC08_frais_procedures apparaît au rang 3.

Cet exemple illustre une dégradation possible
du classement avec des chunks plus courts.

### 7.4 Performance temporelle

Lors de l'exécution MLflow :

- A : 0.0270 ms par recherche FAISS.
- B : 0.0489 ms par recherche FAISS.

Une exécution antérieure avait donné l'ordre inverse.

Ces différences sont trop petites et instables
pour conclure à un avantage de vitesse robuste.

Il faudrait effectuer plusieurs répétitions indépendantes,
mesurer la distribution des latences et inclure
le calcul des embeddings pour évaluer la performance réelle.

## 8. Traçabilité MLflow

Nom de l'expérience :

university_rag_chunking_comparison

Configuration A :

Run ID :
4c3a5c3f4f8a47c9a76994775be965c9

Configuration B :

Run ID :
8474f873138b445eacaabcbe98e982e5

Les exécutions contiennent :

- Les paramètres de chunking.
- Le modèle d'embeddings.
- Le type d'index FAISS.
- Le nombre de documents et de questions.
- Les métriques Hit@1, Recall@5 et MRR@5.
- Les temps mesurés.
- Des empreintes SHA-256 du corpus et des questions.
- Un artefact JSON détaillant les résultats par question.

L'interface de suivi est disponible localement
à l'adresse :

http://127.0.0.1:5000

## 9. Limites expérimentales

Cette expérimentation présente plusieurs limites :

1. Le corpus contient seulement huit documents.
2. Le jeu d'évaluation comporte seulement huit questions.
3. Les références attendues doivent encore être validées
   manuellement dans un protocole d'annotation.
4. Les métriques sont évaluées au niveau du document,
   pas au niveau exact des passages.
5. Une seule famille de modèles d'embeddings est utilisée.
6. L'expérimentation ne compare pas les réponses générées.
7. Les latences FAISS sont très faibles et bruitées.
8. Les documents sont synthétiques et ne représentent
   pas nécessairement des corpus universitaires réels.

Les résultats ne permettent pas de généraliser la supériorité
de la configuration A à d'autres corpus.

## 10. Décision expérimentale

Sur le jeu exploratoire actuel, la configuration A
(180 mots, overlap 35) donne le meilleur classement
des documents pertinents.

Elle est conservée provisoirement comme configuration
de référence.

La configuration B ne sera pas adoptée sur la seule
base de cette expérimentation.

La décision devra être réévaluée avec un jeu de questions
plus large et des métriques de qualité des réponses.

## 11. Reproductibilité

Activer l'environnement Python du projet.

Comparer le découpage :

python -m scripts.compare_chunking

Comparer le retrieval :

python -m scripts.compare_retrieval_configs

Démarrer le serveur MLflow local, puis définir :

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Enregistrer les deux expériences :

python -m scripts.log_retrieval_experiments

Les commandes sont exécutées depuis la racine du projet.

## 12. Conclusion

La comparaison met en évidence l'importance du choix
de la taille des chunks dans un système RAG.

Sur le corpus et les huit questions examinés,
la configuration de référence A obtient de meilleurs
résultats Hit@1 et MRR@5 que la configuration B.

Le suivi MLflow fournit une trace des paramètres,
métriques et résultats de chaque exécution.

L'étape suivante consistera à construire un jeu
d'évaluation annoté plus représentatif et à mesurer
la qualité du retrieval et des réponses générées.

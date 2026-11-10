# Expérimentations et suivi MLflow

## 1. Objectif expérimental

Évaluer la qualité de la recherche documentaire, la génération des réponses et l'effet des modifications de configuration du système RAG.

Les expériences sont suivies avec MLflow.

## 2. Configuration de référence

| Paramètre | Valeur |
|---|---|
| Corpus | 8 documents fictifs |
| Nombre de chunks | 18 |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Index | FAISS IndexFlatIP |
| Retrieval Top-K | 5 |
| Generation Top-K | 3 |
| Seuil de similarité | 0,25 |
| LLM principal | Gemini 3.5 Flash-Lite |

## 3. Évaluation initiale du retrieval

Résultats observés sur huit questions :

| Métrique | Résultat |
|---|---:|
| Recall@5 | 100 % |
| Precision@5 | 32,5 % |
| MRR@5 | 1,00 |
| Latence moyenne de recherche | Environ 20,54 ms |

L'évaluation utilise des documents attendus comme références.

Un document retrouvé ne garantit pas que le passage précis contenant la réponse est présent parmi les chunks transmis au LLM.

## 4. Première comparaison de modèles

Une comparaison exploratoire a été réalisée entre :

- Qwen 2.5 3B avec Ollama.
- Llama 3.2 3B avec Ollama.
- Gemini 3.5 Flash-Lite.

Les modèles ont été testés sur un petit ensemble commun de quatre questions.

Les conditions d'exécution diffèrent entre les modèles locaux sur CPU et le modèle cloud. Les résultats de latence ne constituent donc pas une comparaison contrôlée des seules architectures des modèles.

Cette première expérience a motivé le choix de Gemini pour le prototype.

## 5. Expérience A — RAG baseline

Nom MLflow : `gemini-generation-baseline`

Run ID : `df5207e9a73a4296a7c99d50dbd0182a`

| Métrique | Valeur |
|---|---:|
| Questions évaluées | 10 |
| Taux de réussite fonctionnelle | 90 % |
| Refusal accuracy | 90 % |
| Source hit rate | 87,5 % |
| Latence moyenne | 3,212 s |
| Latence médiane | 1,637 s |

Le principal échec concernait la question :

« Comment obtenir une attestation de scolarité ? »

Le passage contenant la procédure était présent dans le corpus, mais absent des premiers chunks transmis à Gemini.

## 6. Expérience B — Expansion contextuelle

Nom MLflow : `gemini-generation-improved-context`

Run ID : `48ada00d6cfb457f8e47a31593a953f7`

Modification : ajout des chunks voisins des meilleurs résultats FAISS.

| Métrique | Valeur |
|---|---:|
| Questions évaluées | 10 |
| Taux de réussite fonctionnelle | 100 % |
| Refusal accuracy | 100 % |
| Source hit rate | 100 % |
| Latence moyenne | 9,769 s |
| Latence médiane | 1,628 s |
| Latence maximale | 59,689 s |

Les dix tests ont réussi, y compris la question concernant l'attestation.

## 7. Comparaison des configurations

| Métrique | Baseline | Expansion |
|---|---:|---:|
| Réussite fonctionnelle | 90 % | 100 % |
| Refus corrects | 90 % | 100 % |
| Source hit rate | 87,5 % | 100 % |
| Latence médiane | 1,637 s | 1,628 s |
| Latence moyenne | 3,212 s | 9,769 s |

La nouvelle configuration améliore les contrôles fonctionnels sur ce jeu de tests.

La latence moyenne augmente principalement à cause d'une requête ayant pris près de 60 secondes.

La variabilité de l'API Gemini empêche d'attribuer cette différence de latence à la seule expansion contextuelle.

## 8. Analyse des limites

Les contrôles actuels vérifient principalement :

- La présence ou l'absence d'un refus.
- La correspondance entre les identifiants de sources cités et les sources attendues.
- La réussite fonctionnelle sur les questions définies.

Ils ne mesurent pas encore exhaustivement :

- La fidélité de chaque affirmation au contexte.
- La complétude sémantique des réponses.
- La justification précise de chaque citation.
- La résistance aux contradictions.
- La robustesse face aux attaques documentaires.

Un incident HTTP 503 de Gemini a également été observé lors d'une évaluation. Le script d'évaluation a été amélioré pour conserver la progression et permettre une reprise.

## 9. Expériences complémentaires prévues

Afin de consolider l'étude expérimentale, comparer de manière contrôlée :

- Plusieurs stratégies de sélection ou d'expansion du contexte.
- Différentes tailles de chunks et valeurs de chevauchement, si elles sont expérimentées.
- Différentes valeurs de Top-K.
- Les performances avec et sans expansion contextuelle.

Les valeurs 120, 180 et 250 mots, ainsi que les chevauchements 20, 35 et 50 mots, restent des configurations candidates ; elles n'ont pas encore été validées expérimentalement.

Pour chaque comparaison, conserver un jeu de test constant, les paramètres, les résultats et les limitations.

## 10. Métriques futures et quality gates

Ajouter :

- Faithfulness.
- Answer Relevancy.
- Answer Completeness.
- Citation Accuracy.
- Latence P95.
- Taux d'erreur API.
- Tests de refus hors périmètre.

Les seuils d'acceptation devront être définis et justifiés avant leur application en CI.

## 11. Reproductibilité

Code GitHub : commit de référence `f359b80`.

Les exécutions MLflow sont actuellement enregistrées sur un serveur local.

Un export non sensible des métriques, des paramètres et des résultats devra être conservé dans les livrables pour éviter de dépendre uniquement de cette instance locale.

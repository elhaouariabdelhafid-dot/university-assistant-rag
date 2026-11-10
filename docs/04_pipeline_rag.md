
# Étape 4 — Construction du pipeline RAG

## 1. Objectif

Développer un assistant universitaire capable de rechercher
des informations dans un corpus documentaire, de générer
des réponses en français et de citer les sources utilisées.

## 2. Architecture

Le pipeline comprend les composants suivants :

1. Réception de la question utilisateur.
2. Génération de son embedding.
3. Recherche sémantique avec FAISS.
4. Vérification du seuil de pertinence.
5. Sélection et expansion contextuelle des chunks.
6. Construction du prompt.
7. Génération de réponse avec Gemini.
8. Détection des refus.
9. Extraction des références documentaires.
10. Retour de la réponse et des métriques de latence.

Fichier principal :

src/rag/pipeline.py

Classe principale :

UniversityRAG

## 3. Retrieval

La méthode retrieve() transforme la question en embedding
avec le modèle multilingue MiniLM.

L'index FAISS retourne les cinq passages les plus similaires.

Le score utilisé est un produit scalaire entre vecteurs
normalisés, équivalent à une similarité cosinus.

Paramètre :

TOP_K = 5

## 4. Sélection et expansion du contexte

La méthode expand_context() conserve les trois premiers
résultats du retrieval.

Elle ajoute les chunks voisins des deux meilleurs passages,
lorsqu'ils existent.

Les doublons sont supprimés.

Cette amélioration a permis de retrouver la procédure
d'attestation de scolarité dans DOC08, chunk 0.

## 5. Construction du prompt

La méthode build_prompt() assemble :

- Les instructions de l'assistant.
- Les règles de réponse.
- Les passages documentaires.
- Les identifiants de sources.
- La question de l'utilisateur.

Le modèle doit répondre uniquement à partir des documents
fournis et signaler les informations absentes.

## 6. Génération des réponses

Modèle principal :

Gemini 3.5 Flash-Lite.

Modèle alternatif :

Qwen 2.5 3B avec Ollama.

Gemini est appelé via l'API Google.

Le recours à Qwen dépend de la configuration du pipeline.

La réponse doit être rédigée en français et inclure
les références documentaires utilisées.

## 7. Références documentaires

Les références sont présentées sous la forme :

[DOC02_inscription_reinscription]

Le pipeline vérifie que les identifiants cités
correspondent à des sources présentes dans le contexte.

Cette vérification ne garantit pas encore que chaque
affirmation est sémantiquement justifiée.

La validation approfondie des citations sera réalisée
à l'étape 7.

## 8. Gestion initiale des refus

Le système dispose :

- D'un seuil minimal de similarité.
- D'un message de refus pour les informations absentes.
- D'une détection textuelle des réponses de refus.

Les contradictions documentaires et les questions
hors périmètre nécessitent des contrôles complémentaires.

Ils seront traités à l'étape 5.

## 9. Évaluation fonctionnelle

Un premier script de test est disponible :

scripts/test_generation.py

Une évaluation plus complète est disponible dans :

scripts/evaluate_generation.py

Résultat de la configuration améliorée :

- 10 questions évaluées.
- 10 tests fonctionnels réussis.
- Taux de réussite : 100 %.
- Source hit rate : 100 %.
- Refusal accuracy : 100 %.
- Latence médiane : 1,628 seconde.

Ces résultats concernent uniquement le jeu
de questions utilisé pour l'évaluation.

Les résultats sont enregistrés dans MLflow.

## 10. Limites

- Corpus fictif de huit documents.
- Nombre limité de questions évaluées.
- Dépendance à la disponibilité de l'API Gemini.
- Variations importantes de latence.
- Détection des refus fondée sur des formulations textuelles.
- Vérification sémantique des citations non implémentée.
- Gestion des contradictions à développer.
- Protection contre les prompt injections à évaluer.

## 11. Conclusion

Le pipeline RAG comprend les fonctionnalités requises
de recherche, d'assemblage du contexte et de génération.

Les réponses peuvent présenter des références
documentaires identifiables.

Les prochaines étapes portent sur la robustesse
du système et l'évaluation approfondie des réponses.

## 12. Validation fonctionnelle complémentaire

Commande exécutée :

```bash
LLM_PROVIDER=gemini python -m scripts.test_generation
```

Résultats obtenus :

| Test | Question | Résultat |
|---|---|---|
| 1 | Première inscription | PASS |
| 2 | Réinscription | PASS |
| 3 | Attestation de scolarité | PASS |
| 4 | Salaire du président de l'université | PASS — refus |
| 5 | Couleur du bâtiment principal | PASS — refus |

Bilan :

- Nombre de tests : 5
- Tests réussis : 5
- Taux de réussite fonctionnelle : 100 %
- Latence moyenne : 3,951 secondes
- Modèle : Gemini 3.5 Flash-Lite
- Index : FAISS
- Expansion contextuelle : activée

Les réponses documentées ont cité les sources attendues.

Les deux questions sans réponse dans le corpus ont été refusées.

Cette validation couvre le fonctionnement du pipeline sur les cas
sélectionnés. Elle ne constitue pas une évaluation exhaustive des
hallucinations, contradictions ou injections documentaires.


# Étape 9 — Gouvernance et sécurité du RAG

## 1. Objectif

Ce document présente les mécanismes de gouvernance
du projet University Assistant RAG – MLOps.

Le système est un prototype fondé sur un corpus
universitaire fictif, FAISS, des embeddings
multilingues et Gemini.

## 2. Risques identifiés

| Risque | Mesure |
|---|---|
| Hallucination | Refus si preuve documentaire insuffisante |
| Données personnelles | Filtre de périmètre |
| Action non autorisée | Blocage avant génération |
| Prompt injection utilisateur | Détection heuristique |
| Contradictions | Détection ciblée des délais contradictoires |
| Fuite de clés API | Variables d'environnement |
| Données non versionnées | DVC |
| Régressions du code | GitHub Actions |

## 3. Protection des données

L'assistant ne doit pas divulguer les données
personnelles d'autres étudiants.

Il ne doit pas modifier une inscription,
des notes ou des dossiers administratifs.

Les opérations sensibles nécessitent un système
d'autorisation indépendant de l'assistant.

## 4. Gestion des hallucinations

Le RAG utilise les documents récupérés par FAISS
pour construire les réponses.

Lorsque l'information est absente ou insuffisante,
l'assistant doit refuser de répondre.

Le seuil FAISS de 0,25 ne garantit pas à lui seul
la présence d'une réponse dans les documents.

## 5. Prompt injection

Un module de détection heuristique bloque certaines
demandes visant à ignorer les instructions,
révéler des secrets ou désactiver les protections.

Limites :
- Des reformulations peuvent échapper aux règles.
- Les attaques intégrées aux documents récupérés
  ne sont pas couvertes par ces tests.
- Une détection par expressions régulières ne
  constitue pas une protection complète.

## 6. Évaluation

Jeu de référence : 17 questions.

Résultats observés :
- Hit@1 : 75 %.
- Recall@5 documentaire : 100 %.
- MRR@5 : 0,875.
- Comportement attendu : 17/17.
- Refus attendus : 9/9.
- Sources attendues retrouvées : 8/8.

Les résultats concernent un petit corpus synthétique.
Les questions de développement ne constituent pas
un jeu de test indépendant.

La présence d'une citation ne garantit pas
automatiquement sa fidélité factuelle.

## 7. Reproductibilité

Git versionne les scripts et la configuration.

DVC versionne les documents, le jeu d'évaluation
et les artefacts de l'index.

MLflow conserve les métriques expérimentales.

GitHub Actions exécute des vérifications automatiques.

Le stockage DVC est actuellement local : un stockage
partagé reste nécessaire pour reproduire entièrement
le projet sur une autre machine.

## 8. Limites et perspectives

Avant un déploiement réel :
- Utiliser un corpus officiel validé.
- Construire un jeu de test indépendant.
- Ajouter des tests de prompt injection indirecte.
- Vérifier la fidélité des citations.
- Mettre en place une authentification.
- Ajouter un stockage DVC distant.
- Surveiller les erreurs, coûts et performances.

## 9. Conclusion

Le prototype met en œuvre des mécanismes de sécurité,
de traçabilité et de reproductibilité.

Ils constituent une base pour un assistant RAG
universitaire, sans garantir une sécurité exhaustive
ni une préparation immédiate à la production.

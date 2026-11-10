# Charte projet — University Assistant RAG – MLOps

## 1. Problématique

Les informations universitaires concernant les inscriptions, les examens, les formations, les stages et les procédures administratives peuvent être dispersées dans plusieurs documents.

Les étudiants doivent rechercher et interpréter manuellement ces informations, ce qui peut entraîner des pertes de temps et des incompréhensions.

## 2. Objectif du projet

Concevoir un assistant universitaire basé sur une architecture Retrieval-Augmented Generation (RAG), capable de :

- Comprendre une question formulée en langage naturel.
- Rechercher les passages pertinents dans un corpus documentaire.
- Générer une réponse claire en français.
- Présenter les références documentaires utilisées.
- Refuser de répondre lorsque les documents ne contiennent pas l'information demandée.
- Suivre les performances et les expériences avec MLflow.

Le projet constitue un prototype académique et non un service officiel d'une université réelle.

## 3. Utilisateurs cibles

| Utilisateur | Besoin |
|---|---|
| Étudiant | Obtenir des informations administratives et pédagogiques |
| Nouvel étudiant | Comprendre les procédures d'inscription |
| Étudiant en fin de formation | Consulter les procédures de stage et de PFE |
| Personnel administratif | Consulter rapidement les procédures documentées |

## 4. Périmètre fonctionnel

L'assistant peut répondre aux questions documentées concernant :

- Présentation de l'université.
- Première inscription et réinscription.
- Organisation des examens.
- Absences et justificatifs.
- Stages et projets de fin d'études.
- Programmes de formation.
- Calendrier universitaire.
- Frais et procédures administratives.

Le corpus actuel comprend huit documents fictifs de l'Université Al Amal, utilisés uniquement pour le développement et l'évaluation.

## 5. Questions hors périmètre

L'assistant ne doit pas :

- Accéder aux dossiers personnels des étudiants.
- Communiquer des informations confidentielles.
- Inventer des procédures ou des règlements.
- Prendre des décisions administratives.
- Garantir une admission ou la validation d'un diplôme.
- Répondre avec certitude à une question non couverte par le corpus.
- Présenter les documents fictifs comme des règles universitaires officielles.

## 6. Conséquences d'une réponse incorrecte

| Risque | Conséquence potentielle |
|---|---|
| Mauvaise date d'inscription | Inscription tardive |
| Mauvais délai de justification d'absence | Justificatif potentiellement rejeté |
| Liste incomplète de pièces administratives | Dossier incomplet |
| Mauvaise information sur les examens | Mauvaise préparation de l'étudiant |
| Réponse inventée | Perte de confiance dans le système |
| Divulgation d'informations privées | Atteinte à la confidentialité |

Ces conséquences sont des scénarios de risques à prévenir ; elles ne décrivent pas des incidents réellement observés.

## 7. Indicateurs de performance

- Recall@K, Precision@K et MRR pour le retrieval.
- Taux de réussite fonctionnelle des réponses.
- Exactitude des refus.
- Correspondance entre sources citées et sources attendues.
- Faithfulness et complétude des réponses.
- Latence moyenne, médiane et P95.
- Taux d'erreur et disponibilité des services.

## 8. État actuel du prototype

- Corpus : 8 documents universitaires fictifs.
- Index : 18 chunks FAISS.
- Embeddings : modèle multilingue MiniLM.
- LLM principal : Gemini 3.5 Flash-Lite.
- LLM local optionnel : Qwen 2.5 3B avec Ollama.
- Retrieval : FAISS avec expansion contextuelle.
- Évaluation fonctionnelle : 10 tests réussis sur 10.
- Suivi des expériences : MLflow.
- Versionnement du code : Git et GitHub.

Les résultats obtenus sont limités au jeu de tests actuel et ne garantissent pas la fiabilité sur toutes les questions.

## 9. Critères d'acceptation

Le projet devra démontrer que :

1. Le corpus et sa provenance sont documentés.
2. L'index peut être reconstruit de manière reproductible.
3. Les réponses documentées présentent des références vérifiables.
4. Les questions hors corpus produisent un refus approprié.
5. Les contradictions et les limitations sont signalées.
6. Au moins deux configurations sont comparées.
7. Les métriques et les erreurs sont analysées.
8. Les tests automatisés passent les quality gates définis.
9. L'application est exécutable et documentée.
10. Aucun secret ni donnée personnelle n'est publié dans GitHub.

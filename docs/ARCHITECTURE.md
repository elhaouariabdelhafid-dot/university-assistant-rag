# Architecture technique — University Assistant RAG – MLOps

## 1. Vue générale

Le système utilise une architecture Retrieval-Augmented Generation (RAG).

```text
             CORPUS DOCUMENTAIRE
              8 documents fictifs
                      |
                      v
            Extraction / nettoyage
                      |
                      v
             Découpage en chunks
                      |
                      v
        Embeddings multilingues MiniLM
                      |
                      v
                Index FAISS
                      |
                      |
QUESTION UTILISATEUR  |
          |           |
          v           |
   Embedding question |
          |           |
          +-----------+
                      |
                      v
              Retrieval Top-K
                      |
                      v
          Vérification de pertinence
                      |
                      v
          Expansion contextuelle
          (chunks voisins pertinents)
                      |
                      v
            Assemblage du prompt
                      |
                      v
           Gemini 3.5 Flash-Lite
                      |
                      v
          Réponse + sources / refus
                      |
                      v
             Évaluation MLflow
```

FastAPI, Streamlit et le déploiement conteneurisé sont prévus pour une phase ultérieure.

## 2. Corpus documentaire

Le prototype utilise huit documents fictifs représentant différentes procédures universitaires.

Les documents sont traités localement et leur contenu textuel est préparé pour la recherche.

## 3. Modèle d'embeddings

Modèle utilisé :

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Les embeddings sont normalisés pour permettre la recherche par similarité avec FAISS.

Le système est actuellement exécuté sur CPU.

## 4. Index vectoriel

- Moteur : FAISS.
- Type : IndexFlatIP.
- Nombre de chunks : 18.
- Nombre de documents : 8.
- Top-K retrieval par défaut : 5.
- Seuil minimal de pertinence : 0,25.

La similarité utilisée n'est pas une probabilité de vérité.

## 5. Expansion contextuelle

Le pipeline sélectionne les trois meilleurs passages FAISS pour construire le contexte initial.

Il ajoute ensuite les chunks voisins des deux premiers passages lorsque ceux-ci existent et ne sont pas déjà présents.

Cette stratégie a permis de retrouver la procédure d'attestation de scolarité située dans le chunk 0 de DOC08, alors que FAISS classait initialement d'autres passages en tête.

Les chunks ajoutés par voisinage ne possèdent pas nécessairement un score FAISS calculé pour la question.

## 6. Génération des réponses

LLM principal : Gemini 3.5 Flash-Lite, utilisé via une API.

LLM alternatif : Qwen 2.5 3B avec Ollama, utilisable localement.

Le système demande au modèle de :

- Répondre en français.
- Utiliser uniquement le contexte documentaire.
- Citer les sources utilisées.
- Éviter les informations non documentées.
- Refuser lorsqu'une information est absente.

Le modèle local peut être configuré comme fournisseur principal ou secours optionnel. Le secours n'est pas activé automatiquement dans toutes les configurations.

## 7. Gestion des limites

Comportements déjà implémentés :

- Contrôle d'un seuil de similarité.
- Refus des réponses jugées absentes du corpus.
- Extraction des références documentaires présentes dans les réponses.

À renforcer :

- Détection des contradictions documentaires.
- Vérification sémantique des citations.
- Détection des instructions malveillantes dans les documents.
- Gestion robuste des erreurs et indisponibilités API.

## 8. Composants MLOps

| Composant | Technologie | Statut |
|---|---|---|
| Versionnement du code | Git/GitHub | Fonctionnel |
| Suivi des expériences | MLflow | Fonctionnel |
| Évaluation du retrieval | Scripts Python | Fonctionnelle |
| Évaluation des réponses | Scripts Python | Fonctionnelle |
| Versionnement des données | DVC | À intégrer |
| Tests automatisés | Pytest | À compléter |
| Intégration continue | GitHub Actions | À mettre en place |
| API applicative | FastAPI | À développer |
| Interface | Streamlit | À développer |
| Conteneurisation applicative | Docker | À compléter |

## 9. Limites architecturales

- Corpus réduit et fictif.
- Évaluation sur un nombre limité de questions.
- Dépendance à une API externe pour Gemini.
- Latence variable du modèle génératif.
- Absence actuelle d'une vérification sémantique complète des réponses.
- Absence de mécanisme validé de résolution des contradictions.

Ces limites devront être documentées et évaluées avant toute utilisation institutionnelle réelle.

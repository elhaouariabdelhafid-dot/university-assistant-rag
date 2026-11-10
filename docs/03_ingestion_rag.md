
# Étape 3 — Ingestion RAG

## 1. Objectif

Transformer le corpus universitaire en représentations
vectorielles afin de permettre une recherche sémantique
rapide et traçable.

L'ingestion comprend :
- Le chargement des documents TXT.
- Le découpage en chunks.
- La conservation des métadonnées.
- La génération des embeddings.
- La construction de l'index FAISS.

## 2. Corpus utilisé

Le corpus comprend huit documents universitaires fictifs.

Répertoire :
data/processed/text/

Nombre de documents : 8
Nombre de chunks : 18

La provenance des documents et leurs empreintes SHA-256
sont enregistrées dans :

data/metadata/documents.csv

## 3. Stratégie de chunking

Fichier :
src/rag/chunking.py

Stratégie : découpage par mots, avec chevauchement.

Paramètres :
- chunk_size : 180 mots
- chunk_overlap : 35 mots
- step : 145 mots

Le chevauchement permet de conserver une partie
du contexte entre deux passages successifs.

Les fichiers sont parcourus dans un ordre déterministe
grâce à sorted().

Limite :
Le découpage par mots ne respecte pas nécessairement
les frontières des phrases ou des sections.

## 4. Métadonnées des chunks

Chaque chunk contient :

- text : contenu textuel.
- source : identifiant du document source.
- chunk_id : position du chunk dans le document.
- status : statut documentaire, actuellement unknown.

Exemple :

{
  "text": "Contenu documentaire...",
  "source": "DOC08_frais_procedures",
  "chunk_id": 0,
  "status": "unknown"
}

Les métadonnées de version, les dates et les empreintes
SHA-256 sont conservées séparément dans l'inventaire.

Une extension future permettra de relier directement
chaque chunk à sa version documentaire.

## 5. Modèle d'embeddings

Modèle utilisé :

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Caractéristiques :
- Embeddings multilingues.
- Dimension : 384.
- Exécution sur CPU.
- Normalisation des vecteurs activée.

Le modèle est adapté au prototype et permet
de rechercher des passages à partir de requêtes
en langage naturel.

Les performances pour les questions en arabe
ou en darija restent à évaluer.

## 6. Stratégie d'indexation

Bibliothèque : FAISS

Type d'index :
IndexFlatIP

Les embeddings sont normalisés avant indexation.

La similarité est calculée avec un produit scalaire
équivalent à la similarité cosinus pour des vecteurs
normalisés.

L'index conserve les vecteurs des 18 chunks.

## 7. Artefacts générés

Le pipeline d'indexation produit :

artifacts/university.faiss
artifacts/chunks.json

Le fichier FAISS contient les vecteurs.

Le fichier JSON contient les textes et les
métadonnées des chunks.

L'ordre des chunks correspond à l'ordre
des vecteurs insérés dans FAISS.

## 8. Reconstruction de l'index

Commande :

python -m scripts.build_index

Cette commande reconstruit les artefacts à partir
des fichiers TXT présents dans data/processed/text/.

Attention :
La commande remplace les artefacts existants.

Une copie de sauvegarde doit être conservée
avant toute reconstruction expérimentale.

## 9. Validation de reproductibilité

Résultats observés :

- Documents : 8.
- Chunks régénérés : 18.
- Chunks enregistrés : 18.
- Chunks identiques : True.
- Vecteurs FAISS : 18.
- Dimension : 384.
- Norme minimale : 1,0.
- Norme maximale : 1,0.
- Similarité minimale recalculée : 1,0.
- Normalisation : validée.
- Cohérence des vecteurs : validée.

La vérification a été réalisée sans modifier
les artefacts enregistrés.

Elle confirme la cohérence dans l'environnement
logiciel actuel.

## 10. Limites et améliorations futures

- Le chunking par mots peut couper une phrase.
- La taille des chunks n'a pas encore été optimisée.
- Les versions des dépendances doivent être figées.
- Le modèle utilisé à la recherche doit correspondre
  à celui utilisé pour construire l'index.
- Les versions et hashes documentaires devront être
  liés plus directement aux chunks.
- Une comparaison de plusieurs configurations de
  chunking sera effectuée dans les expérimentations.

## 11. Conclusion

Le pipeline d'ingestion RAG est fonctionnel.

Le découpage, les embeddings et l'index FAISS
ont été validés.

L'étape suivante consiste à vérifier le pipeline
de retrieval, d'assemblage du contexte et de génération.

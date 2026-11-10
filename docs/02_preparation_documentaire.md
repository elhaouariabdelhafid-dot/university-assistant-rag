
# Étape 2 — Préparation documentaire

## 1. Objectif

Préparer un corpus documentaire traçable pour un assistant
universitaire basé sur une architecture RAG.

## 2. Nature des données

Le prototype utilise huit documents fictifs associés à
l'Université Al Amal.

Les documents sont destinés exclusivement à
l'expérimentation académique.

Ils ne constituent pas des sources institutionnelles
officielles.

## 3. Corpus disponible

| Document | Nombre de chunks |
|---|---:|
| DOC01 — Présentation | 2 |
| DOC02 — Inscription et réinscription | 2 |
| DOC03 — Examens | 2 |
| DOC04 — Absences et justificatifs | 2 |
| DOC05 — Stages et PFE | 2 |
| DOC06 — Formations | 3 |
| DOC07 — Calendrier universitaire | 2 |
| DOC08 — Frais et procédures | 3 |
| **Total** | **18** |

Les huit documents contiennent au total 2 169 mots,
selon le comptage réalisé lors de l'inventaire.

## 4. Format et extraction

Les documents utilisés sont des fichiers TXT déjà
disponibles dans :

data/processed/text/

Le répertoire data/raw/ est actuellement vide.

Aucune extraction PDF n'est nécessaire pour ce corpus
synthétique existant.

Une procédure d'extraction PDF pourra être ajoutée
pour une extension future du projet.

## 5. Inventaire et provenance

Le script scripts/create_inventory.py produit :

data/metadata/documents.csv

L'inventaire contient notamment :

- Identifiant documentaire
- Nom et chemin du fichier
- Type et origine déclarée
- Empreinte SHA-256
- Taille et nombre de mots
- Statut de qualité technique
- Métadonnées de version et de date, si disponibles

Les informations non vérifiées restent inconnues.

## 6. Contrôles de qualité

Le script scripts/validate_corpus.py vérifie :

- L'existence des documents inventoriés
- L'intégrité SHA-256
- La présence des sources des chunks
- L'unicité des identifiants
- L'absence de chunks vides
- La présence d'au moins un chunk par document

Résultat observé :

- 8 documents validés
- 18 chunks
- Aucune erreur détectée

Ces contrôles ne garantissent pas l'exactitude
sémantique des informations.

## 7. Reproductibilité

Recréer l'inventaire :

python -m scripts.create_inventory

Valider le corpus :

python -m scripts.validate_corpus

Les fichiers sources et les paramètres d'ingestion
doivent être versionnés afin de permettre
la reconstruction de l'index.

## 8. Limites

- Corpus synthétique de petite taille
- Absence de sources PDF brutes pour cette version
- Dates et statuts documentaires non vérifiés
- Contrôles sémantiques à approfondir
- Reproductibilité complète de l'index à vérifier
  à l'étape 3

## 9. Conclusion

Les vérifications techniques initiales de la préparation
documentaire ont été réalisées avec succès.

Le corpus est prêt pour la validation de l'ingestion RAG.

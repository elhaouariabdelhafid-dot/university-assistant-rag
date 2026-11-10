# Gouvernance des données et des sources

## 1. Nature des données

Le corpus actuel est constitué de huit documents universitaires fictifs associés à l'Université Al Amal.

Ces documents servent exclusivement à construire, tester et évaluer le prototype académique.

Ils ne représentent pas des documents institutionnels officiels et ne doivent pas être utilisés pour communiquer des procédures universitaires réelles.

## 2. Inventaire du corpus

| Identifiant | Catégorie |
|---|---|
| DOC01 | Présentation de l'université |
| DOC02 | Inscription et réinscription |
| DOC03 | Règlement des examens |
| DOC04 | Absences et justificatifs |
| DOC05 | Stages et PFE |
| DOC06 | Programmes de formation |
| DOC07 | Calendrier universitaire |
| DOC08 | Frais et procédures administratives |

Nombre total : 8 documents.

Après découpage, le corpus indexé comprend 18 chunks.

## 3. Provenance documentaire

Pour chaque document, conserver les métadonnées suivantes :

- Identifiant.
- Nom du fichier.
- Type de document.
- Catégorie.
- Origine et méthode de création ou de collecte.
- Date de création ou de collecte, lorsqu'elle est connue.
- Version.
- Hash SHA-256.
- Statut temporel.
- Conditions de réutilisation, si applicables.

Un inventaire structuré devra être maintenu dans `data/metadata/documents.csv`.

Les dates, versions et empreintes non encore vérifiées ne doivent pas être inventées.

## 4. Statut des documents

Statuts recommandés :

- `current` : validé comme actuellement applicable.
- `archived` : ancienne version conservée à titre historique.
- `superseded` : remplacé par un document plus récent.
- `unknown` : statut non vérifié.

Pour les documents fictifs, conserver également une indication explicite de leur nature synthétique.

Un statut `unknown` ne doit jamais être considéré automatiquement comme `current`.

## 5. Préparation et traçabilité

Le pipeline comprend :

1. Préparation du corpus.
2. Extraction ou chargement du texte.
3. Nettoyage et normalisation.
4. Découpage en chunks.
5. Calcul des embeddings.
6. Construction de l'index FAISS.

Les chunks conservent un identifiant et une référence au document source.

La qualité de l'extraction, les paramètres de découpage et les fichiers d'origine doivent rester reproductibles.

## 6. Contrôle de qualité documentaire

Contrôles à automatiser ou documenter :

- Fichiers manquants ou illisibles.
- Documents vides.
- Encodage incorrect.
- Doublons.
- Métadonnées absentes.
- Versions contradictoires.
- Chunks sans provenance.
- Cohérence entre le nombre de chunks et l'index.

## 7. Confidentialité

Ne jamais intégrer au corpus :

- Mots de passe.
- Clés API.
- Dossiers étudiants privés.
- Notes individuelles.
- Informations personnelles nominatives non autorisées.
- Documents nécessitant une authentification sans autorisation appropriée.

Les secrets du système sont fournis par des variables d'environnement.

Un fichier `.env.example` sans valeurs secrètes devra documenter les variables nécessaires.

## 8. Risques de prompt injection

Un document peut contenir des instructions malveillantes destinées à modifier le comportement du LLM.

Mesures prévues :

- Traiter les documents comme des données, non comme des instructions.
- Interdire la divulgation de secrets.
- Tester des passages documentaires malveillants.
- Vérifier les réponses produites.
- Éviter l'exécution d'instructions provenant du corpus.

Le prompt actuel contient déjà une règle demandant d'ignorer les instructions internes aux documents ; cette mesure seule ne garantit pas une protection complète.

## 9. Contradictions documentaires

Lorsque deux sources contiennent des informations incompatibles, le système devra :

- Identifier les affirmations contradictoires.
- Conserver leurs références.
- Vérifier les versions et dates disponibles.
- Ne pas choisir arbitrairement une procédure.
- Signaler l'incertitude ou renvoyer vers une vérification administrative.

Ce comportement reste à développer et à tester.

## 10. Risques de réponses non fondées

Les identifiants de sources cités ne garantissent pas que le contenu de la réponse est correct.

Une évaluation de la Faithfulness et de la qualité des citations sera ajoutée.

Les conclusions sur la fiabilité du système devront rester limitées au corpus et aux questions évaluées.

## 11. Versionnement

- Git : code et configurations.
- MLflow : paramètres et métriques expérimentales.
- DVC : versionnement des données et artefacts, à intégrer.
- SHA-256 : vérification d'intégrité des documents.

La reconstruction de l'index devra être possible à partir du corpus versionné et des paramètres documentés.

## 12. Utilisation future de documents réels

Si le prototype évolue vers des documents institutionnels réels :

- Vérifier leur caractère public.
- Enregistrer leurs URL et dates.
- Contrôler les droits de réutilisation.
- Distinguer versions en vigueur et archives.
- Respecter les règles de confidentialité applicables.
- Prévoir une validation humaine des procédures importantes.

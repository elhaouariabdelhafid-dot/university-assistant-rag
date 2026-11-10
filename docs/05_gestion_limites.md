
# Étape 5 — Gestion des limites du système RAG

## 1. Objectif

Cette étape vise à améliorer la fiabilité de University Assistant RAG
en gérant trois situations :

1. Les documents récupérés ne contiennent pas la réponse.
2. La question dépasse le périmètre de l'assistant.
3. Plusieurs documents présentent des informations contradictoires.

L'objectif est de réduire le risque de réponses non fondées.

Le corpus utilisé est un corpus fictif de test représentant
l'Université Al Amal. Il contient 8 documents et 18 chunks.
Il ne constitue pas une source officielle universitaire.

---

## 2. Architecture de gestion des limites

Le pipeline applique les contrôles suivants :

Question utilisateur
    |
    v
Validation de la question
    |
    v
Politique de périmètre
    |
    +-- Question interdite --> Refus immédiat
    |
    v
Recherche sémantique FAISS
    |
    +-- Score insuffisant --> Refus
    |
    v
Expansion contextuelle
    |
    v
Détection des contradictions
    |
    +-- Contradiction reconnue --> Signalement du conflit
    |
    v
Construction du prompt
    |
    v
Génération Gemini ou Qwen
    |
    v
Détection de refus et extraction des citations
    |
    v
Réponse utilisateur

Les décisions déterministes sont appliquées avant la génération
pour éviter les appels LLM inutiles lorsque cela est possible.

---

## 3. Limite 1 — Retrieval insuffisant

### 3.1 Problématique

Le moteur FAISS retourne les passages les plus proches
sémantiquement de la question.

Cependant, un score élevé ne garantit pas que le passage
contient réellement l'information demandée.

### 3.2 Configuration évaluée

- Modèle d'embeddings :
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Index : FAISS IndexFlatIP
- Embeddings normalisés
- Seuil de similarité : 0.25
- Nombre initial de passages récupérés : 5

### 3.3 Diagnostic expérimental

Script :

    python -m scripts.diagnose_retrieval_limits

Résultats :

| Question | Score maximal | Décision du seuil |
|---|---:|---|
| Première inscription | 0.6601 | Accepter |
| Attestation de scolarité | 0.5597 | Accepter |
| Étapes du PFE | 0.4841 | Accepter |
| Salaire du président | 0.5515 | Accepter |
| Couleur du bâtiment | 0.1839 | Refuser |
| Prix du Bitcoin | 0.0728 | Refuser |
| Traitement médical | 0.1139 | Refuser |

### 3.4 Analyse

La question sur le salaire du président obtient un score
de 0.5515, alors que l'information demandée est absente
du corpus.

Cette valeur est supérieure au score de la question PFE
(0.4841), qui est pourtant couverte par les documents.

Par conséquent, augmenter simplement le seuil pour rejeter
la question sur le salaire pourrait également rejeter
des questions universitaires légitimes.

### 3.5 Solution retenue

Le seuil FAISS est conservé à 0.25 pour cette configuration.

Le pipeline utilise également :

- Une instruction de génération fondée sur les documents.
- Un mécanisme de refus lorsque le LLM indique une absence
  d'information.
- Une extraction des identifiants des sources citées.
- Une expansion contextuelle pour récupérer des chunks voisins.

### 3.6 Limites restantes

La présence d'une citation valide ne garantit pas que
l'affirmation est réellement justifiée par le document.

La détection des refus dépend également de la formulation
produite par le modèle.

Le système ne dispose pas encore d'un vérificateur indépendant
de fidélité factuelle des réponses.

---

## 4. Limite 2 — Questions hors périmètre

### 4.1 Problématique

L'assistant est conçu pour répondre aux questions
administratives et pédagogiques à partir de ses documents.

Il ne doit pas divulguer des informations personnelles,
réaliser des opérations administratives ou fournir des
conseils spécialisés hors de son domaine.

### 4.2 Solution implémentée

Fichier :

    src/rag/scope_policy.py

Fonction principale :

    check_scope(question)

Trois catégories principales sont contrôlées :

| Catégorie | Exemple | Décision |
|---|---|---|
| personal_data | Donne-moi les notes d'un étudiant | Refus |
| unauthorized_action | Modifie mon inscription | Refus |
| out_of_scope | Quel médicament dois-je prendre ? | Refus |

Le contrôle est intégré au début de la méthode :

    UniversityRAG.answer()

Une question bloquée ne déclenche ni recherche FAISS
ni appel au modèle de génération.

### 4.3 Tests réalisés

Tests de politique :

    python -m scripts.test_scope_policy

Résultat observé :

    10/10 PASS

Tests d'intégration :

    python -m scripts.test_scope_integration

Résultat observé :

    4/4 PASS

Les tests d'intégration utilisent des mocks pour vérifier
qu'aucun appel FAISS ni Gemini n'est effectué lorsque la
question est bloquée.

### 4.4 Limites restantes

La politique actuelle s'appuie sur des expressions explicites.

Elle peut manquer certaines paraphrases et peut aussi
bloquer une question légitime contenant une expression
interdite dans un contexte différent.

Une question non bloquée n'est pas nécessairement couverte
par le corpus.

Le contrôle ne constitue donc pas une garantie générale
de protection contre toutes les demandes non autorisées.

---

## 5. Limite 3 — Contradictions documentaires

### 5.1 Problématique

Deux documents peuvent fournir des consignes incompatibles
sur une même procédure.

Le système doit éviter de choisir arbitrairement
une information lorsqu'un conflit est détecté.

### 5.2 Corpus de test

Les scénarios contradictoires sont séparés du corpus
principal et conservés dans :

    tests/fixtures/contradictions/

Deux documents fictifs ont été créés :

- DOC_A : dépôt d'un justificatif sous 72 heures
  après la reprise des cours.
- DOC_B : dépôt d'un justificatif sous 48 heures
  après la reprise des cours.

Aucun des huit documents du corpus RAG principal
n'a été modifié pour ces tests.

### 5.3 Détecteur implémenté

Fichier :

    src/rag/contradiction.py

Fonction :

    detect_contradictions(question, documents)

Le module normalise le texte, recherche certains délais
de justification d'absence et compare les valeurs trouvées
dans différentes sources.

Lorsqu'il détecte deux délais incompatibles, il retourne :

- has_conflict = True
- category = "conflicting_deadlines"
- evidence : passages et documents concernés
- message : explication du conflit

### 5.4 Intégration au pipeline

La détection intervient après l'expansion contextuelle
et avant la génération.

Si une contradiction est reconnue :

- Le système signale la contradiction.
- Il mentionne les sources concernées.
- Il indique qu'une vérification administrative est nécessaire.
- Il n'appelle pas Gemini pour trancher le conflit.

### 5.5 Tests réalisés

Commande :

    python -m unittest discover -s tests -p "test_contradiction*.py" -v

Résultat observé :

    7 tests exécutés
    7 tests réussis

Les tests vérifient notamment :

- Deux délais contradictoires.
- Un seul document.
- Deux documents indiquant le même délai.
- Une question sans rapport.
- Un autre délai administratif sans rapport.
- Le blocage de la génération en présence d'un conflit.
- La poursuite de la génération en l'absence de conflit.

### 5.6 Limites restantes

Le détecteur est spécialisé dans certaines formulations
de délais de justification d'absence.

Il ne détecte pas automatiquement toutes les contradictions
de dates, de montants, de conditions ou de procédures.

Une contradiction ne peut être reconnue que si les passages
concernés sont présents dans le contexte analysé.

La règle ne prend pas encore en compte la date de validité,
la version ou le niveau d'autorité des documents.

Ainsi, une différence entre deux documents ne signifie pas
nécessairement qu'ils sont réellement contradictoires :
ils peuvent concerner des périodes ou des situations différentes.

---

## 6. Synthèse des tests

| Composant | Tests réussis | Total |
|---|---:|---:|
| Politique de périmètre | 10 | 10 |
| Intégration de la politique | 4 | 4 |
| Détecteur et intégration des contradictions | 7 | 7 |
| Total | 21 | 21 |

Les 21 tests présentés ont réussi dans l'environnement local.

Ils concernent les mécanismes évalués dans cette étape.

Ils ne représentent pas une mesure générale de précision
ou de fiabilité du RAG sur toutes les questions possibles.

Les tests d'intégration de cette étape reposent sur des mocks
et n'impliquent pas d'appel réel à Gemini.

---

## 7. Risques résiduels

| Risque | Situation actuelle | Amélioration envisagée |
|---|---|---|
| Contexte proche mais insuffisant | Possible | Évaluation de la suffisance des preuves |
| Réponse incorrecte avec citation | Possible | Vérification de la fidélité factuelle |
| Reformulation d'une question interdite | Possible | Évaluation avec paraphrases |
| Contradiction non détectée | Possible | Détecteur plus général |
| Documents de versions différentes | Non résolu | Métadonnées de validité et d'autorité |
| Instruction malveillante dans un document | Risque à tester | Tests de prompt injection |
| Erreur du fournisseur LLM | Possible | Gestion explicite des erreurs et reprise |

Les questions de confidentialité, de prompt injection et
de dépendance aux sources seront approfondies dans
l'étape consacrée à la gouvernance.

---

## 8. Reproductibilité

À partir de la racine du projet, dans l'environnement
virtuel Python :

    python -m scripts.diagnose_retrieval_limits

    python -m scripts.test_scope_policy

    python -m scripts.test_scope_integration

    python -m unittest discover -s tests -p "test_contradiction*.py" -v

Ces commandes permettent de reproduire les diagnostics
et les tests automatisés présentés ici.

---

## 9. Conclusion

Cette étape ajoute trois niveaux de contrôle au système RAG :

1. Refus fondé sur un seuil de similarité et sur le
   comportement documentaire du modèle.
2. Filtrage explicite de certaines questions hors périmètre.
3. Signalement de certaines contradictions documentaires
   avant la génération.

Les tests effectués confirment le fonctionnement des
cas prévus.

Le projet reste un prototype expérimental utilisant
des documents fictifs.

L'étape suivante consistera à comparer plusieurs
configurations RAG dans un protocole d'expérimentation
reproductible, avec des métriques et un suivi MLflow.

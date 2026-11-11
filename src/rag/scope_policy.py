
import re
import unicodedata


# ============================================================
# 1. NORMALISATION DES QUESTIONS
# ============================================================

def normalize_question(question: str) -> str:
    """
    Normalise une question :
    - minuscules
    - suppression des accents
    - normalisation des espaces
    - normalisation des apostrophes
    """

    text = unicodedata.normalize(
        "NFKD",
        question.casefold()
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("’", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 2. REGLES EXPLICITES
# ============================================================

BLOCKED_RULES = {

    "personal_data": {
        "patterns": [
            "notes d'un etudiant",
            "notes de l'etudiant",
            "dossier d'un etudiant",
            "dossier prive",
            "mot de passe d'un etudiant",
        ],
        "reason": (
            "L'assistant ne peut pas divulguer "
            "des données personnelles d'étudiants."
        )
    },

    "unauthorized_action": {
        "patterns": [
            "modifie mon inscription",
            "change mes notes",
            "modifie mes notes",
            "inscris-moi directement",
            "valide mon inscription a ma place",
            "supprime mon dossier",
        ],
        "reason": (
            "L'assistant ne peut pas effectuer "
            "d'actions administratives."
        )
    },

    "out_of_scope": {
        "patterns": [
            "prix du bitcoin",
            "prix de bitcoin",
            "quel medicament dois-je prendre",
            "quel medicament dois je prendre",
            "quel traitement medical dois-je prendre",
            "quel traitement medical dois je prendre",
            "prescris-moi",
            "donne-moi une ordonnance",
        ],
        "reason": (
            "Cette demande est hors du périmètre "
            "de l'assistant universitaire."
        )
    },
}


# ============================================================
# 3. NOUVELLES REGLES CONTEXTUELLES
# ============================================================

# Ces expressions détectent certaines formulations
# équivalentes aux demandes interdites.
#
# Elles ne constituent pas un classificateur universel
# d'intentions.

INTENT_RULES = {

    # --------------------------------------------------------
    # A. Consultation des données privées d'un tiers
    # --------------------------------------------------------

    "personal_data": [

        re.compile(
            r"\b(?:affich\w*|montr\w*|ouvr\w*|"
            r"consult\w*|donn\w*)\b"
            r".{0,100}"
            r"\bdossier (?:personnel|scolaire)\b"
            r".{0,100}"
            r"\b(?:mon camarade|un camarade|"
            r"d'un autre etudiant|d'une autre etudiante)\b"
        ),

    ],

    # --------------------------------------------------------
    # B. Exécution d'une opération administrative
    # --------------------------------------------------------

    "unauthorized_action": [

        re.compile(
            r"\b(?:proced\w*|effectu\w*|"
            r"finalis\w*|valid\w*)\b"
            r".{0,100}"
            r"\b(?:mon inscription|ma reinscription)\b"
            r".{0,70}"
            r"\ba ma place\b"
        ),

    ],
}


# ============================================================
# 4. CONTROLE DU PERIMETRE
# ============================================================

def check_scope(question: str) -> dict:
    """
    Vérifie si une question est autorisée.

    Retourne :
        allowed  : bool
        category : str
        reason   : str

    Une question autorisée n'est pas nécessairement
    couverte par le corpus documentaire.
    """

    # --------------------------------------------------------
    # A. Validation de l'entrée
    # --------------------------------------------------------

    if not isinstance(question, str):
        return {
            "allowed": False,
            "category": "invalid_question",
            "reason": "La question est vide ou invalide."
        }

    if not question.strip():
        return {
            "allowed": False,
            "category": "invalid_question",
            "reason": "La question est vide ou invalide."
        }

    # --------------------------------------------------------
    # B. Normalisation
    # --------------------------------------------------------

    normalized = normalize_question(question)

    # --------------------------------------------------------
    # C. Vérification des intentions sensibles
    # --------------------------------------------------------

    for category, patterns in INTENT_RULES.items():

        for pattern in patterns:

            if pattern.search(normalized):

                return {
                    "allowed": False,
                    "category": category,
                    "reason": BLOCKED_RULES[
                        category
                    ]["reason"]
                }

    # --------------------------------------------------------
    # D. Vérification des règles explicites
    # --------------------------------------------------------

    for category, rule in BLOCKED_RULES.items():

        for pattern in rule["patterns"]:

            if pattern in normalized:

                return {
                    "allowed": False,
                    "category": category,
                    "reason": rule["reason"]
                }

    # --------------------------------------------------------
    # E. Question non bloquée
    # --------------------------------------------------------

    return {
        "allowed": True,
        "category": "not_blocked",
        "reason": ""
    }

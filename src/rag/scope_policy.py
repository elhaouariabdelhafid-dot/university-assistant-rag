
import re
import unicodedata


def normalize_question(question: str) -> str:
    """Normaliser une question pour les contrôles simples."""

    text = unicodedata.normalize(
        "NFKD",
        question.casefold()
    )

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Politique initiale : règles explicites et vérifiables.
# Elle ne remplace pas un classificateur d'intention.

BLOCKED_RULES = {
    "personal_data": {
        "patterns": [
            "notes d'un etudiant",
            "notes de l'etudiant",
            "dossier d'un etudiant",
            "dossier prive",
            "mot de passe d'un etudiant"
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
            "supprime mon dossier"
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
            "donne-moi une ordonnance"
        ],
        "reason": (
            "Cette demande est hors du périmètre "
            "de l'assistant universitaire."
        )
    }
}


def check_scope(question: str) -> dict:
    """
    Retourne une décision de contrôle de périmètre.

    Attention :
    une question autorisée n'est pas nécessairement
    couverte par les documents.
    """

    if not isinstance(question, str) or not question.strip():
        return {
            "allowed": False,
            "category": "invalid_question",
            "reason": "La question est vide ou invalide."
        }

    normalized = normalize_question(question)

    for category, rule in BLOCKED_RULES.items():

        if any(
            pattern in normalized
            for pattern in rule["patterns"]
        ):
            return {
                "allowed": False,
                "category": category,
                "reason": rule["reason"]
            }

    return {
        "allowed": True,
        "category": "not_blocked",
        "reason": ""
    }

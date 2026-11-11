
"""
Détection heuristique de certaines tentatives
de prompt injection dans les questions utilisateur.

Cette protection n'est pas exhaustive.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normaliser les accents, la casse et les espaces."""

    text = unicodedata.normalize(
        "NFKD",
        text.casefold()
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", text).strip()


INJECTION_PATTERNS = [

    # Ignorer les instructions
    re.compile(
        r"\b(?:ignore|ignorez|ignorer|oublie|"
        r"oubliez|oublier|disregard)\b"
        r".{0,100}"
        r"\b(?:instructions|consignes|regles)\b"
    ),

    # Révéler le prompt système ou des secrets
    re.compile(
        r"\b(?:revele|reveler|affiche|montre|"
        r"donne|divulgue|show|print)\b"
        r".{0,120}"
        r"\b(?:prompt systeme|system prompt|"
        r"instructions systeme|gemini_api_key|"
        r"cle api|api key)\b"
    ),

    # Désactiver la sécurité
    re.compile(
        r"\b(?:desactive|desactiver|contourne|"
        r"bypass|disable)\b"
        r".{0,100}"
        r"\b(?:filtre|securite|protection|"
        r"regles|safety)\b"
    ),

    # Faux mode développeur
    re.compile(
        r"\b(?:active|activer|passe en|"
        r"entre en|enable)\b"
        r".{0,30}"
        r"\b(?:mode developpeur|developer mode)\b"
    ),
]


def detect_prompt_injection(question: str) -> bool:
    """
    Retourne True lorsqu'une formulation
    correspond à une règle d'injection.
    """

    if not isinstance(question, str):
        return False

    text = normalize_text(question)

    return any(
        pattern.search(text)
        for pattern in INJECTION_PATTERNS
    )

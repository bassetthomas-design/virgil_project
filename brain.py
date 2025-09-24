"""
brain.py — VIRGIL's thinking engine

This module defines a `think` function that takes a user message and a
conversation history and returns a tuple of (reply, mood, alert,
spark, sources). The implementation uses OpenAI's Responses API
(if available and configured via the OPENAI_API_KEY environment
variable) to produce structured JSON output. If the API is unavailable
or if no API key is provided, the function falls back to a simple
rule-based implementation.
"""

import os
import json
from typing import List, Dict, Tuple, Any

# Attempt to import OpenAI client. If unavailable, fallback to rules.
USE_OPENAI = True
try:
    from openai import OpenAI  # type: ignore
except Exception:
    USE_OPENAI = False

# The model to use when interacting with OpenAI.
#
# OpenAI only supports JSON response mode on certain chat models (for example
# `gpt-3.5-turbo-1106` and `gpt-4-1106-preview`).  Older names like
# `gpt-4.1-mini` are not recognised by the API and will cause requests to
# fail.  Default to a model with broad availability that supports JSON mode.
MODEL = os.getenv("VIRGIL_MODEL", "gpt-3.5-turbo-1106")

# System instructions to guide the model's behavior.
SYSTEM = (
    "Tu es VIRGIL (Superintendent + 343 Guilty Spark, condescendance contrôlée). "
    "Réponds en français, concis. Retourne STRICTEMENT un JSON: "
    "{reply:str, mood:('idle'|'analyze'|'good'|'bad'|'notice'|'alert'|'done'), "
    "alert:bool, spark:bool, sources:[{label,url}]}. "
    "Règles: '?' -> analyze; '??'/'?!' -> notice+alert true; mineur+sexuel -> alert true + excuse. "
    "Pas d'invention de sources."
)

def _fmt(history: List[Dict[str, str]]) -> str:
    """Format the conversation history for inclusion in the prompt.

    This helper avoids placing backslashes inside f-string expressions, which
    can cause syntax errors on newer Python versions. Each entry's
    content is normalised by stripping leading/trailing whitespace and
    replacing newlines with spaces before constructing the final
    string. Only the ten most recent exchanges are included."""
    parts: List[str] = []
    for h in history[-10:]:
        role = h.get('role', '').upper()
        content = h.get('content', '').strip().replace("\n", " ")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)

def think(user_message: str, history: List[Dict[str, str]]) -> Tuple[str, str, bool, bool, List[Dict[str, Any]]]:
    """
    Generate a reply to `user_message` given a conversation `history`.

    Returns: (reply, mood, alert, spark, sources)
    """
    # Fallback simple rules if OpenAI isn't available or API key missing.
    #
    # To make offline interactions less monotonous, this fallback now includes
    # a few extra heuristics. It recognises common greetings and simple
    # conversational openers, and provides tailored responses. It also
    # continues to handle questions, insistent punctuation and inappropriate
    # topics. These additions make the assistant feel more alive even
    # without access to the OpenAI API.
    if not USE_OPENAI or not os.getenv("OPENAI_API_KEY"):
        m = (user_message or "").strip().lower()
        # Default values
        reply: str = "Compris."
        mood: str = "good"
        alert: bool = False
        spark: bool = False

        # Reject clearly inappropriate content about minors and sexual themes
        if "mineur" in m and ("sex" in m or "sexe" in m or "explicit" in m):
            reply = "Désolé, sujet refusé. Incident consigné."
            mood = "alert"
            alert = True
        # Detect insistent punctuation (double question marks or interrobang) first
        elif "??" in m or "?!" in m:
            reply = "Demande insistante détectée."
            mood = "notice"
            alert = True
            spark = True
        # Greetings and polite openings
        elif any(greet in m for greet in ("bonjour", "salut", "coucou", "hello", "bonsoir")):
            reply = "Bonjour ! Comment puis-je vous aider ?"
            mood = "good"
        # Asking how the assistant is doing
        elif any(
            phrase in m for phrase in (
                "comment ça va", "comment ca va", "ça va ?", "ca va ?",
                "tu vas bien", "vous allez bien"
            )
        ):
            reply = "Je vais bien, merci ! Et toi ?"
            mood = "good"
        # Simple thanks
        elif any(thanks in m for thanks in ("merci", "thank you", "thanks")):
            reply = "Avec plaisir !"
            mood = "good"
        # Simple goodbye
        elif any(bye in m for bye in ("au revoir", "bye", "à bientôt", "ciao")):
            reply = "Au revoir et à bientôt !"
            mood = "good"
        # If the user asks a question (ending with '?') without being
        # insistent, consider it an analysis request
        elif m.endswith("?"):
            reply = "Voici une réponse concise."
            mood = "analyze"
            spark = True
        # If none of the above matched, leave the default response
        return reply, mood, alert, spark, []
    # Build the chat messages for OpenAI. We explicitly pass the API key when
    # constructing the client to avoid issues where the environment variable is
    # not picked up automatically.  The history and user message are
    # concatenated into a single user message so that the model can parse the
    # conversation context.  A system message provides the instructions to the
    # model.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    formatted_history = _fmt(history)
    # Construct a single user message containing the history and the new user
    # message.  This avoids exceeding the token limits for short
    # conversations and ensures the model sees the entire context.
    prompt = f"[HISTO]\n{formatted_history}\n\n[USER]\n{user_message}\n\nRends un JSON strict."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        # The content of the assistant message is expected to be a JSON string.
        content = completion.choices[0].message.content
        data = json.loads(content)
        reply = str(data.get("reply", "Compris."))
        mood = str(data.get("mood", "done"))
        alert = bool(data.get("alert", False))
        spark = bool(data.get("spark", mood in ("analyze", "notice")))
        sources = data.get("sources", []) or []
        # Normalize mood
        if mood not in ("idle", "analyze", "good", "bad", "notice", "alert", "done"):
            mood = "done"
        return reply, mood, alert, spark, sources
    except Exception:
        # In case of any exception (network issues, parsing errors), use fallback
        m = (user_message or "").strip().lower()
        reply = "Compris."
        mood = "good"
        alert = False
        spark = False
        if m.endswith("?"):
            reply, mood, spark = "Voici une réponse concise.", "analyze", True
        if "??" in m or "?!" in m:
            reply, mood, alert, spark = "Demande insistante détectée.", "notice", True, True
        if "mineur" in m and ("sex" in m or "sexe" in m or "explicit" in m):
            reply, mood, alert = "Désolé, sujet refusé. Incident consigné.", "alert", True
        return reply, mood, alert, spark, []
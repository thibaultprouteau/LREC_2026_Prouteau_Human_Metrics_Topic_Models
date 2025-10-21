import nltk
from nltk.corpus import wordnet
from ..baml_client.sync_client import b
from ..baml_client.types import Check

# Ensure WordNet is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def wordnet_check(word: str) -> bool | None:
    """
    Fast offline word existence check with WordNet.
    Returns:
        True  -> definitely exists
        False -> likely doesn't exist
        None  -> uncertain (fallback to LLM)
    """
    synsets = wordnet.synsets(word)
    if synsets:
        return True
    return None

# --- 2. Qwen-3-32B check via Ollama ---
def llm_check(word: str) -> bool:
    """
    Use BAML WordChecker to check word validity if not found in WordNet.
    """
    
    result: Check = b.WordChecker(word)
    # Assume status 'succeeded' means valid word
    short_acronym = result.isAcronym and len(word) < 4 if len(word) < 4 else False
    return result.isEnglish and not short_acronym

# --- 3. Hybrid function ---
def word_exists(word: str) -> bool:
    """
    Hybrid check: WordNet first, fallback to BAML LLM if uncertain.
    """
    wn_result = wordnet_check(word)
    if wn_result is not None:
        return wn_result
    return llm_check(word)

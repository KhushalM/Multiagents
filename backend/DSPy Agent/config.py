import os
from typing import Optional

from dotenv import load_dotenv  # type: ignore
import dspy as dspy  # type: ignore


def init_lm(model_name: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.2):
    """Initialize DSPy's LM with sensible defaults.

    By default, we use OpenAI-compatible provider via environment.
    The user can override model via `model_name` or environment `DSPY_MODEL`.
    """
    # Load environment variables from a .env file if present
    load_dotenv()

    resolved_model = model_name or os.getenv("DSPY_MODEL", "gpt-5")

    # OpenAI-style LM through DSPy. You can swap providers (e.g., Azure, Groq) if needed.
    # DSPy picks provider from environment by default; this sets the model string and args.
    lm = dspy.LM(model=resolved_model, api_key=os.getenv(api_key_env), temperature=temperature)
    # Configure DSPy globally
    dspy.configure(lm=lm)
    return lm


def init_retriever(k: int = 5):
    """Return a simple in-memory TopK retriever placeholder to be wired by our RAG module.

    For Wikipedia we won't build a vector index initially; we'll fetch on demand and rank.
    """
    # We'll implement retrieval logic inside the module; this function is a placeholder for symmetry.
    return {"top_k": k}



"""Every adapter that makes paid calls. A structural test checks each one requires a SpendGuard.

Kept out of app/providers/__init__.py: importing the adapters there would create an import cycle
(app.cost.budget imports app.providers.errors, which runs the package __init__).
"""

from app.providers.asr.gemini import GeminiASRProvider
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.tts.gemini import GeminiTTSProvider

PAID_ADAPTERS: tuple[type, ...] = (GeminiLLMProvider, GeminiTTSProvider, GeminiASRProvider)

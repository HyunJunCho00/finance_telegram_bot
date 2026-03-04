import sys
import os
sys.path.append(os.getcwd())

from config.settings import settings

print(f"GROQ_API_KEY present: {bool(settings.GROQ_API_KEY)}")
print(f"CEREBRAS_API_KEY present: {bool(settings.CEREBRAS_API_KEY)}")
print(f"GEMINI_API_KEY present: {bool(settings.GEMINI_API_KEY)}")
print(f"GEMINI_API_KEY_JUDGE present: {bool(settings.GEMINI_API_KEY_JUDGE)}")
print(f"GEMINI_API_KEY_VLM present: {bool(settings.GEMINI_API_KEY_VLM)}")

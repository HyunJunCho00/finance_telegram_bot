import sys
import os
import traceback
sys.path.append(os.getcwd())

from agents.ai_router import ai_client
from config.settings import settings

def test_model(role, name):
    print(f"\n--- Testing {name} ---")
    try:
        # Use a very simple prompt
        backend, model_id, cap = ai_client._get_route(role)
        print(f"Routing: backend={backend}, model_id={model_id}")
        
        # We'll call the backend-specific generator directly to avoid the fallback logic in generate_response for better debugging
        if backend == "cerebras":
            result = ai_client._generate_openai_compat(
                ai_client.cerebras_client, model_id, 
                "You are a helpful assistant.", "Say hi", 
                10, 0.7, None, name="Cerebras"
            )
        elif backend == "groq":
             result = ai_client._generate_openai_compat(
                ai_client.groq_client, model_id, 
                "You are a helpful assistant.", "Say hi", 
                10, 0.7, None, name="Groq"
            )
        else:
            print(f"Skipping backend {backend}")
            return

        print(f"Result: '{result}'")
        if not result:
            print(f"FAILED: Empty response from {name}")
    except Exception as e:
        print(f"EXCEPTION in {name}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_model("meta_regime", "Cerebras")
    test_model("rag_extraction", "Groq")

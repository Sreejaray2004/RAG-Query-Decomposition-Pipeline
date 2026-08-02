import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# 1. Load variables from .env
load_dotenv()

# --- CONFIGURE THESE ---
# You can hardcode your repo_id here to test, e.g., "meta-llama/Llama-3.1-8B-Instruct"
LLM_REPO_ID = os.getenv("LLM_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
# -----------------------

print(f"Testing model: {LLM_REPO_ID}")
print(f"Token found: {'Yes (starts with ' + HF_TOKEN[:4] + '...)' if HF_TOKEN else 'NO - Token is empty!'}")
print("-" * 50)

if not HF_TOKEN:
    print("❌ Error: No token found. Please set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in your .env file.")
    exit(1)

try:
    # 2. Initialize the client targeting your specific model
    client = InferenceClient(model=LLM_REPO_ID, token=HF_TOKEN)

    # 3. Attempt a minimal chat completion call
    print("Sending request to Hugging Face router...")
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=10,
    )

    reply = response.choices[0].message.content
    print("✅ SUCCESS! Your token has the correct permissions.")
    print(f"Model response: {reply}")

except HfHubHTTPError as e:
    print("\n❌ HTTP ERROR: Hugging Face rejected the request.")
    print(f"Status Code: {e.response.status_code}")
    print(f"Details: {e}")
    
    if e.response.status_code == 403:
        print("\n--> DIAGNOSIS: Your token lacks 'Inference API' permissions or you need to accept the model's license agreement on huggingface.co.")
    elif e.response.status_code == 404:
        print("\n--> DIAGNOSIS: The LLM_REPO_ID was not found or is not supported by the free serverless Inference API.")
        
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
import os
from langchain_openai import AzureChatOpenAI

# Load environment variables
api_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_LANG = "en-US"  # Default fallback

AZURE_SPEECH_VOICE_1 = "en-US-Emma2:DragonHDLatestNeural"
AZURE_SPEECH_VOICE_2 = "en-US-Andrew3:DragonHDLatestNeural"

tenant_id = os.getenv("AZURE_TENANT_ID")
client_id = os.getenv("AZURE_CLIENT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")

OUTPUT_FILENAME = "podcast.wav"  # Default, will be timestamped later

# Global LLM
global_llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    azure_deployment=deployment_name,
    api_version=api_version,
    temperature=1
)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp"
}

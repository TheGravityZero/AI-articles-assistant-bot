import logging
import os

# Log configuration
logger = logging.getLogger("ai_articles_assistant")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Telegram bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    logging.error(
        "BOT_TOKEN env var is not found"
    )
else:
    logging.info("BOT_TOKEN found, starting the bot")

# Hugging face token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logging.error(
        "HF_TOKEN env var is not found"
    )
else:
    logging.info("HF_TOKEN found")

# Pinecon token
PINECONE_TOKEN = os.getenv("PINECONE_TOKEN")
if not PINECONE_TOKEN:
    logging.error(
        "PINECONE_TOKEN env var is not found"
    )
else:
    logging.info("PINECONE_TOKEN found")


# Model configuration
MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"
logging.info(f"MODEL_ID is {MODEL_ID}")

INDEX_ID = "llama-2-rag"
logging.info(f"INDEX_ID is {INDEX_ID}")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
logging.info(f"EMBED_MODEL_ID is {EMBED_MODEL_ID}")

PINECONE_ENVIRONMENT_ID = "gcp-starter"
logging.info(f"PINECONE_ENVIRONMENT_ID is {PINECONE_ENVIRONMENT_ID}")


# Bot inference settings
INFERENCE_SETTINGS = {
    'use_rag': False,
}

# Model initialization settings
MODEL_INIT_SETTINGS = {
    'model_id': MODEL_ID,
    'index_id': INDEX_ID,
    'embed_model_id': EMBED_MODEL_ID,
    'pinecone_environment_id': PINECONE_ENVIRONMENT_ID,
}

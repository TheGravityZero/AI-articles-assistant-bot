# AI-articles-assistant-bot

**Installation**
```
git clone https://github.com/TheGravityZero/AI-articles-assistant-bot
cd AI-articles-assistant-bot
pip install -qUr requirements.txt
```

**Launch bot**
```
python3 main.py
```

**Environment variables**  
_(add them with ```export```)_
```
BOT_TOKEN - telegram bot token from BotFather
HF_TOKEN - HuggingFace token
PINECONE_TOKEN - Pinecone token
```

**Bot usage**
```
/disable_rag - Disable usage of RAG (disabled by default)
/enable_rag - Enable usage of RAG
/help - Get help
```
Type any question and bot will reply with an answer

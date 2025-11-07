import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Atlas Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'chatbot_db')
    
    # Collections - Use your existing collection name or leave as default
    # If you have existing collections, specify the name in .env file
    EXISTING_COLLECTION_NAME = os.getenv('EXISTING_COLLECTION_NAME', None)  # Your existing collection
    COLLECTION_DATA = os.getenv('EXISTING_COLLECTION_NAME', 'dataset')  # Use existing or default
    COLLECTION_CHAT_HISTORY = 'chat_history'  # For storing chat conversations
    COLLECTION_EMBEDDINGS = 'embeddings'  # For storing vector embeddings for RAG
    
    # OpenAI Configuration (for chatbot)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # OpenRouter Configuration (alternative free models)
    # Get your API key from: https://openrouter.ai/keys
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    # Free models available: meta-llama/llama-3.2-3b-instruct:free, google/gemma-2-2b-it:free, microsoft/phi-3-mini-128k-instruct:free
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.2-3b-instruct:free')
    OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
    
    # Server Configuration
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    # Web Research
    ALLOW_WEB = os.getenv('ALLOW_WEB', 'True').lower() == 'true'


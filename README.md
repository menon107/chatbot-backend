# MongoDB Atlas Chatbot

A real-time chatbot that uses your existing MongoDB Atlas database to answer questions based on your data.

## Features

- ✅ Connects to your existing MongoDB Atlas database
- ✅ Uses your existing collections and data
- ✅ Real-time data ingestion
- ✅ RAG (Retrieval-Augmented Generation) for intelligent responses
- ✅ Chat history management
- ✅ Flexible data retrieval that adapts to your schema

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure MongoDB Atlas Connection

1. Create a `.env` file in the project root (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your MongoDB Atlas credentials:
   ```env
   MONGODB_URI=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/?retryWrites=true&w=majority
   DATABASE_NAME=your_database_name
   EXISTING_COLLECTION_NAME=your_collection_name
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   **Important:** 
   - Replace `your_username` and `your_password` with your MongoDB Atlas credentials
   - Replace `your_cluster` with your actual cluster name
   - Replace `your_database_name` with your existing database name
   - Replace `your_collection_name` with your existing collection name (optional)
   - Get your OpenAI API key from https://platform.openai.com/api-keys

### 3. Get MongoDB Atlas Connection String

1. Log in to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Go to your cluster → Click "Connect"
3. Choose "Connect your application"
4. Copy the connection string
5. Replace `<password>` with your database user password
6. Paste it into your `.env` file as `MONGODB_URI`

### 4. Whitelist Your IP Address

1. In MongoDB Atlas, go to "Network Access"
2. Click "Add IP Address"
3. Add your current IP or `0.0.0.0/0` for development (not recommended for production)

### 5. Test Connection

Run the setup script to verify your connection:

```bash
python setup_database.py
```

This will:
- Test your MongoDB connection
- Show available collections
- Display sample document structure

### 6. Train on Your Existing Data

Process your existing MongoDB data and create indexes:

```bash
python train_on_existing_db.py
```

Or train on a specific collection:

```bash
python train_on_existing_db.py your_collection_name
```

This will:
- Create text search indexes on your data
- Prepare your data for efficient querying
- Show statistics about your collections

### 7. Start the Chatbot Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Chat with the Bot
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Your question here",
  "session_id": "optional_session_id"
}
```

### Ingest Real-time Data
```bash
POST /api/ingest
Content-Type: application/json

{
  "data_type": "news",
  "content": "Your data content here",
  "metadata": {}
}
```

### Get Conversation History
```bash
GET /api/history/<session_id>
```

### Get Latest Data
```bash
GET /api/data/latest?limit=10&data_type=news
```

## Frontend (Coming Soon)

A web interface will be available at `http://localhost:5000` after starting the server.

## How It Works

1. **Data Retrieval**: The chatbot searches your MongoDB collections using text search
2. **Context Extraction**: Relevant documents are retrieved based on the user's query
3. **RAG Generation**: The retrieved context is combined with the user's question
4. **Response Generation**: OpenAI GPT generates an intelligent response based on your data
5. **History Storage**: All conversations are stored in MongoDB for context

## Troubleshooting

### Connection Issues

- **"ServerSelectionTimeoutError"**: Check your IP whitelist in MongoDB Atlas
- **"Authentication failed"**: Verify your username and password in the connection string
- **"Network error"**: Check your internet connection and firewall settings

### Search Not Working

- Run `train_on_existing_db.py` to create text indexes
- Make sure your documents have text fields (not just numbers or IDs)

### No Data Found

- Verify your collection name in `.env` matches your actual collection
- Check that your collection has documents
- Run `setup_database.py` to see available collections

## Security Notes

- Never commit your `.env` file to version control
- Use strong passwords for MongoDB Atlas
- Restrict IP whitelist in production
- Rotate API keys regularly

## Support

For issues or questions, check:
- MongoDB Atlas documentation: https://docs.atlas.mongodb.com/
- OpenAI API documentation: https://platform.openai.com/docs/


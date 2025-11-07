# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure MongoDB Atlas Connection

1. **Create a `.env` file** (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. **Get your MongoDB Atlas connection string**:
   - Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
   - Log in to your account
   - Click on your cluster → "Connect"
   - Choose "Connect your application"
   - Copy the connection string

3. **Edit `.env` file** and add:
   ```env
   MONGODB_URI=mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@YOUR_CLUSTER.mongodb.net/?retryWrites=true&w=majority
   DATABASE_NAME=your_existing_database_name
   EXISTING_COLLECTION_NAME=your_existing_collection_name
   OPENAI_API_KEY=your_openai_api_key
   ```

   **Important**: 
   - Replace `YOUR_USERNAME` and `YOUR_PASSWORD` with your actual MongoDB credentials
   - Replace `YOUR_CLUSTER` with your cluster name
   - Replace `your_existing_database_name` with your database name
   - Replace `your_existing_collection_name` with your collection name (optional)
   - Get OpenAI API key from: https://platform.openai.com/api-keys

4. **Whitelist your IP** in MongoDB Atlas:
   - Go to "Network Access" in MongoDB Atlas
   - Click "Add IP Address"
   - Add your current IP or `0.0.0.0/0` for development

## Step 3: Test Connection

```bash
python setup_database.py
```

This will verify your connection and show your collections.

## Step 4: Train on Your Existing Data

```bash
python train_on_existing_db.py
```

This processes your existing MongoDB data and creates indexes for efficient searching.

## Step 5: Start the Chatbot Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## Step 6: Open the Chatbot Interface

Open your browser and go to:
```
http://localhost:5000
```

You can now chat with your bot! It will use your existing MongoDB data to answer questions.

---

## Troubleshooting

### Connection Errors
- Check your `.env` file has correct credentials
- Verify IP whitelist in MongoDB Atlas
- Make sure your password doesn't have special characters (if it does, URL-encode them)

### No Data Found
- Run `setup_database.py` to see available collections
- Make sure `EXISTING_COLLECTION_NAME` in `.env` matches your collection name
- Run `train_on_existing_db.py` to process your data

### API Errors
- Make sure OpenAI API key is valid
- Check your internet connection

---

## What You Need

✅ **MongoDB Atlas Account** - Free tier works fine  
✅ **MongoDB Atlas Database** - Your existing database with data  
✅ **OpenAI API Key** - For intelligent chatbot responses  
✅ **Python 3.7+** - To run the application

---

That's it! Your chatbot is ready to use your MongoDB Atlas data! 🚀


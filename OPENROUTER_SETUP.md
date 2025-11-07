# OpenRouter Setup Guide

OpenRouter provides access to free AI models that you can use with your chatbot!

## Step 1: Get Your OpenRouter API Key

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up for a free account
3. Go to [API Keys](https://openrouter.ai/keys)
4. Create a new API key
5. Copy your API key

## Step 2: Configure Your .env File

Add these lines to your `.env` file:

```env
# OpenRouter Configuration (Free Models)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free
```

## Step 3: Available Free Models

You can use any of these free models:

1. **meta-llama/llama-3.2-3b-instruct:free** (Recommended)
   - Fast and efficient
   - Good for general conversations
   - Default model

2. **google/gemma-2-2b-it:free**
   - Google's Gemma model
   - Good performance

3. **microsoft/phi-3-mini-128k-instruct:free**
   - Microsoft's Phi-3 model
   - Good for technical questions

To use a different model, just change `OPENROUTER_MODEL` in your `.env` file.

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the `requests` library needed for OpenRouter.

## Step 5: Restart Your Server

```bash
python app.py
```

## How It Works

The chatbot will now:
1. **Try OpenRouter first** (free models)
2. **Fall back to OpenAI** (if OpenRouter fails and OpenAI is configured)
3. **Use fallback mode** (if both fail, uses MongoDB data directly)

## Priority Order

1. OpenRouter (if API key configured) ← **Free!**
2. OpenAI (if API key configured)
3. Fallback mode (uses MongoDB data directly)

## Benefits of OpenRouter

✅ **Free models available**  
✅ **No credit card required**  
✅ **Multiple model options**  
✅ **Easy to set up**  
✅ **Automatic fallback**  

## Troubleshooting

**If OpenRouter doesn't work:**
- Check your API key is correct
- Make sure you've installed `requests`: `pip install requests`
- Check the model name is correct (see available models above)
- Check your internet connection

**If you get rate limit errors:**
- OpenRouter has rate limits on free models
- The chatbot will automatically fall back to OpenAI or fallback mode
- Wait a few minutes and try again

## Example .env Configuration

```env
# MongoDB
MONGODB_URI=mongodb+srv://...
DATABASE_NAME=your_database

# OpenRouter (Free - Priority)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free

# OpenAI (Optional - Fallback)
OPENAI_API_KEY=sk-...

# Server
FLASK_PORT=5000
```

That's it! Your chatbot will now use free AI models from OpenRouter! 🎉


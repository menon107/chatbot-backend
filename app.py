"""
Flask API server for the chatbot
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from chatbot import Chatbot
from data_ingestion import DataIngestion
from config import Config
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
FRONT_DIST_DIR = os.path.join(BASE_DIR, 'front', 'dist')
FRONT_BUILD_DIR = os.path.join(BASE_DIR, 'front', 'build')  # legacy
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')  # user's app root
FRONTEND_DIST = os.path.join(FRONTEND_DIR, 'dist')  # built output for Vite apps
FRONTEND_DIST_SPA = os.path.join(FRONTEND_DIST, 'spa')  # some templates emit dist/spa

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize services
chatbot = Chatbot()
data_ingestion = DataIngestion()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "chatbot-api"})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - main chatbot interaction"""
    try:
        data = request.get_json()
        user_query = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        patient_email = data.get('email', None)  # Get patient email from request
        
        if not user_query:
            return jsonify({"error": "Message is required"}), 400
        
        # Generate response with patient email filter
        response = chatbot.generate_response(user_query, session_id, patient_email=patient_email)
        
        return jsonify({
            "response": response,
            "session_id": session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest_data():
    """Endpoint for real-time data ingestion"""
    try:
        data = request.get_json()
        data_type = data.get('data_type', 'general')
        content = data.get('content', '')
        metadata = data.get('metadata', {})
        
        if not content:
            return jsonify({"error": "Content is required"}), 400
        
        # Store data
        document_id = data_ingestion.store_data(data_type, content, metadata)
        
        return jsonify({
            "message": "Data ingested successfully",
            "document_id": str(document_id)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get conversation history for a session"""
    try:
        history = chatbot.get_conversation_history(session_id)
        return jsonify({
            "session_id": session_id,
            "history": history
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/latest', methods=['GET'])
def get_latest_data():
    """Get latest data updates"""
    try:
        limit = request.args.get('limit', 10, type=int)
        data_type = request.args.get('data_type', None)
        
        data = data_ingestion.get_latest_data(limit=limit, data_type=data_type)
        
        # Convert ObjectId to string for JSON serialization
        for item in data:
            item['_id'] = str(item['_id'])
            if 'timestamp' in item:
                item['timestamp'] = item['timestamp'].isoformat()
        
        return jsonify({
            "data": data,
            "count": len(data)
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving latest data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/name', methods=['GET'])
def get_profile_name():
    """Return the latest patient's profile name only (if available)."""
    try:
        # Prefer documents that contain a profile.name, most recent first
        doc = chatbot.data_retrieval.collection.find({"profile.name": {"$exists": True}}).sort("updatedAt", -1).limit(1)
        docs = list(doc)
        if not docs:
            # Fallback: any profile doc by createdAt
            doc = chatbot.data_retrieval.collection.find({"profile.name": {"$exists": True}}).sort("createdAt", -1).limit(1)
            docs = list(doc)
        if docs:
            name = docs[0].get("profile", {}).get("name")
            return jsonify({"name": name}), 200
        return jsonify({"name": None}), 200
    except Exception as e:
        logger.error(f"Error retrieving profile name: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/email', methods=['GET'])
def get_profile_email():
    """Return the latest patient's email from the users collection."""
    try:
        # Get the most recent document with a profile and email, sorted by updatedAt
        doc = chatbot.data_retrieval.collection.find({"email": {"$exists": True}}).sort("updatedAt", -1).limit(1)
        docs = list(doc)
        if not docs:
            # Fallback: sort by createdAt
            doc = chatbot.data_retrieval.collection.find({"email": {"$exists": True}}).sort("createdAt", -1).limit(1)
            docs = list(doc)
        if docs:
            email = docs[0].get("email")
            if email:
                return jsonify({"email": email}), 200
        return jsonify({"email": None}), 200
    except Exception as e:
        logger.error(f"Error retrieving patient email: {e}")
        return jsonify({"error": str(e)}), 500

def _active_front_dir():
    """Prefer built 'frontend/dist' if present, else raw 'frontend', else legacy 'front' builds."""
    # Serve built Vite app if available
    if os.path.isfile(os.path.join(FRONTEND_DIST_SPA, 'index.html')):
        return FRONTEND_DIST_SPA
    if os.path.isfile(os.path.join(FRONTEND_DIST, 'index.html')):
        return FRONTEND_DIST
    # Otherwise, try raw frontend directory (for plain static sites)
    if os.path.isfile(os.path.join(FRONTEND_DIR, 'index.html')):
        return FRONTEND_DIR
    # Fallback to legacy front build
    if os.path.isfile(os.path.join(FRONT_DIST_DIR, 'index.html')):
        return FRONT_DIST_DIR
    if os.path.isfile(os.path.join(FRONT_BUILD_DIR, 'index.html')):
        return FRONT_BUILD_DIR
    return None

@app.route('/assets/<path:filename>')
def assets(filename):
    """Serve asset files from the active frontend directory (frontend/assets or build assets)."""
    active = _active_front_dir()
    if not active:
        return jsonify({"error": "Frontend not found."}), 404
    assets_dir = os.path.join(active, 'assets')
    if os.path.isdir(assets_dir):
        return send_from_directory(assets_dir, filename)
    # Try direct file under active dir
    file_path = os.path.join(active, filename)
    if os.path.isfile(file_path):
        return send_from_directory(active, filename)
    return jsonify({"error": "Asset not found."}), 404

@app.route('/')
def index():
    """Serve the user's frontend app (frontend/index.html)."""
    active = _active_front_dir()
    if active:
        return send_from_directory(active, 'index.html')
    return "Frontend not found", 404

# Serve any other static files from the active frontend directory (for Vite dev-like structure)
@app.route('/<path:path>')
def static_files(path):
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    active = _active_front_dir()
    if active:
        candidate = os.path.join(active, path)
        if os.path.isfile(candidate):
            directory = os.path.dirname(candidate)
            filename = os.path.basename(candidate)
            return send_from_directory(directory, filename)
        # SPA fallback
        return send_from_directory(active, 'index.html')
    return "Frontend not found", 404

if __name__ == '__main__':
    logger.info(f"Starting Flask server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )


"""
Chatbot with RAG (Retrieval-Augmented Generation) using MongoDB data
"""
import os
from typing import List, Dict
from datetime import datetime
from database import db
from config import Config
from data_ingestion import DataIngestion
from data_retrieval import DataRetrieval
from date_parser import extract_date_from_query
from query_parser import extract_health_metric, extract_profile_field, extract_time_context, parse_query
import logging

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available. Install requests package for OpenRouter and web research support.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install openai package for full functionality.")

class Chatbot:
    def __init__(self, collection_name=None):
        """
        Initialize chatbot
        
        Args:
            collection_name: Name of your existing MongoDB collection. If None, uses config.
        """
        self.data_ingestion = DataIngestion()
        self.data_retrieval = DataRetrieval(collection_name)
        self.chat_history_collection = db.get_collection(Config.COLLECTION_CHAT_HISTORY)
        self.data_collection = self.data_retrieval.collection
        
        # Initialize OpenRouter (priority - free models)
        if REQUESTS_AVAILABLE and Config.OPENROUTER_API_KEY:
            self.use_openrouter = True
            self.openrouter_api_key = Config.OPENROUTER_API_KEY
            self.openrouter_model = Config.OPENROUTER_MODEL
            self.openrouter_base_url = Config.OPENROUTER_BASE_URL
            logger.info(f"OpenRouter configured with model: {self.openrouter_model}")
        else:
            self.use_openrouter = False
            logger.info("OpenRouter not configured. Using fallback options.")
        
        # Initialize OpenAI (fallback)
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.use_openai = True
            logger.info("OpenAI configured as fallback.")
        else:
            self.use_openai = False
            if not self.use_openrouter:
                logger.warning("No AI models configured. Using basic retrieval mode.")
    
    # -------------------- Stats and Recommendation Utilities --------------------
    def _parse_stat_request(self, user_query: str):
        """Extract metric, statistic type, and day window from the user's query.
        Returns (metric_key, stat_type, days) or (None, None, None)."""
        import re
        q = user_query.lower()

        stat_aliases = {
            'average': 'mean', 'avg': 'mean', 'mean': 'mean',
            'median': 'median', 'mode': 'mode',
            'min': 'min', 'minimum': 'min', 'lowest': 'min',
            'max': 'max', 'maximum': 'max', 'highest': 'max',
            'std': 'std', 'stdev': 'std', 'standard deviation': 'std',
        }
        stat_type = None
        for alias, normalized in stat_aliases.items():
            if alias in q:
                stat_type = normalized
                break

        metric_map = {
            'systolic': 'systolic', 'diastolic': 'diastolic', 'pulse': 'pulse',
            'bmi': 'bmi', 'glucose': 'fastingBloodSugar', 'blood sugar': 'fastingBloodSugar',
            'sugar': 'fastingBloodSugar', 'cholesterol': 'totalCholesterol',
            'waist': 'waistCircumference', 'sleep': 'sleepQuality', 'stress': 'stressLevel',
        }
        metric_key = None
        for phrase, key in metric_map.items():
            if phrase in q:
                metric_key = key
                break

        days = None
        m = re.search(r"last\s+(\d+)\s+days|past\s+(\d+)\s+days", q)
        if m:
            days = int(next(g for g in m.groups() if g))
        else:
            m2 = re.search(r"for\s+(\d+)\s+days", q)
            if m2:
                days = int(m2.group(1))

        return metric_key, stat_type, days

    def _compute_metric_stats(self, bp_readings: list, metric: str, days: int = None):
        """Compute stats over bp_readings for a metric, optionally limited to last N days."""
        from datetime import datetime, timedelta
        from statistics import mean, median
        from collections import Counter

        def parse_date(d):
            try:
                return datetime.strptime(d, '%Y-%m-%d')
            except Exception:
                return None

        now = datetime.utcnow()
        cutoff = now - timedelta(days=days) if days else None

        values = []
        for r in bp_readings:
            if metric not in r or r.get(metric) is None:
                continue
            if cutoff is not None and r.get('date'):
                dt = parse_date(r.get('date'))
                if dt and dt < cutoff:
                    continue
            values.append(r.get(metric))

        if not values:
            return None

        def to_float(v):
            if isinstance(v, dict):
                try:
                    v = list(v.values())[0]
                except Exception:
                    pass
            try:
                return float(v)
            except Exception:
                return None

        nums = [to_float(v) for v in values]
        nums = [n for n in nums if n is not None]
        if not nums:
            return None

        nums_sorted = sorted(nums)
        counter = Counter(nums_sorted)
        max_freq = max(counter.values())
        modes = sorted([v for v, c in counter.items() if c == max_freq])

        try:
            mval = mean(nums)
            variance = sum((x - mval) ** 2 for x in nums) / len(nums)
            std = variance ** 0.5
        except Exception:
            std = None

        return {
            'count': len(nums),
            'min': nums_sorted[0],
            'max': nums_sorted[-1],
            'mean': sum(nums) / len(nums),
            'median': median(nums),
            'mode': modes[0] if modes else None,
            'std': std,
        }

    def _unit_for_metric(self, metric: str) -> str:
        return {
            'systolic': 'mmHg', 'diastolic': 'mmHg', 'pulse': 'bpm',
            'fastingBloodSugar': 'mg/dL', 'totalCholesterol': 'mg/dL',
            'waistCircumference': 'cm', 'bmi': '', 'sleepQuality': '/5', 'stressLevel': '/5',
        }.get(metric, '')

    def _format_bullet_points(self, text: str) -> str:
        """
        Format bullet points in text: ensure each bullet (except first) starts on a new line,
        and add a blank line after each bullet point.
        
        Uses move_bullets_to_newline to handle multiple bullet types and exclude the first bullet.
        
        Args:
            text: The text to format
            
        Returns:
            Formatted text with proper bullet point line breaks
        """
        import re
        
        if not text:
            return text
        
        BULLET_PATTERN = re.compile(r'(-|\*|•|–|—|\d+[.)])')
        
        def move_bullets_to_newline(text: str) -> str:
            """
            Insert a newline before every bullet marker found in `text`, except:
              - if the bullet is already at the start of a line, do nothing for that bullet,
              - skip inserting for the first bullet encountered (the "exclude the first bullet" rule).
            Recognized bullet markers: -, *, •, –, —, and numbered bullets like "1." or "2)".
            """
            if not text:
                return text
            
            out_parts = []
            last_index = 0
            bullets_seen = 0
            
            # Iterate over matches in order
            for m in BULLET_PATTERN.finditer(text):
                start = m.start()
                
                # If this is the very first bullet encountered in the entire text,
                # we skip transformation (we exclude the first bullet).
                if bullets_seen == 0:
                    bullets_seen += 1
                    continue
                
                # If bullet is already at line start, no insertion needed.
                # A bullet is at line start if 'start == 0' or previous character is newline.
                if start == 0 or text[start - 1] == '\n':
                    # Nothing to insert; continue scanning
                    continue
                
                # Insert text between last_index and this start, then add a newline,
                # then continue from start (we do not consume the bullet itself).
                out_parts.append(text[last_index:start])
                out_parts.append('\n')          # ensure the bullet starts on a new line
                last_index = start
            
            # append whatever remains
            out_parts.append(text[last_index:])
            return ''.join(out_parts)
        
        # First, move bullets to new lines (except first)
        text = move_bullets_to_newline(text)
        
        # Now add blank lines after each bullet point
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Preserve blank lines but don't add extra
                if formatted_lines and formatted_lines[-1] != '':
                    formatted_lines.append('')
                continue
            
            # Check if line starts with a bullet point
            bullet_match = re.match(r'^[•\-\*–—]|\d+[.)]', stripped)
            if bullet_match:
                # Bullet line - add it and then add a blank line after
                formatted_lines.append(stripped)
                formatted_lines.append('')
            else:
                # Non-bullet line
                formatted_lines.append(stripped)
        
        # Join lines
        result = '\n'.join(formatted_lines)
        
        # Clean up: ensure max 2 consecutive newlines (one blank line between bullets)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Remove trailing blank lines
        result = result.rstrip()
        
        return result

    # Removed hardcoded recommendations - model will generate based on evidence

    def _extract_metric_value(self, user_query: str):
        """Try to pull a (metric, numeric value) pair from free-text like 'systolic 150' or 'bp is 160'."""
        import re
        q = user_query.lower()
        metric_aliases = {
            'systolic': 'systolic', 'diastolic': 'diastolic', 'pulse': 'pulse',
            'heart rate': 'pulse', 'bmi': 'bmi',
            'glucose': 'fastingBloodSugar', 'blood sugar': 'fastingBloodSugar', 'sugar': 'fastingBloodSugar',
            'cholesterol': 'totalCholesterol', 'ldl': 'totalCholesterol',
            'waist': 'waistCircumference', 'sleep': 'sleepQuality', 'stress': 'stressLevel'
        }
        # numeric first
        for phrase, key in metric_aliases.items():
            if phrase in q:
                m = re.search(r"(\d+\.?\d*)", q)
                if m:
                    try:
                        return key, float(m.group(1))
                    except Exception:
                        pass
        return None, None
    
    def _extract_patient_identifier(self, query: str) -> dict:
        """
        Extract patient name or email from query
        
        Returns:
            dict with 'name' and/or 'email' if found
        """
        import re
        
        patient_info = {}
        query_lower = query.lower()
        
        # Extract email pattern
        email_pattern = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
        email_match = re.search(email_pattern, query)
        if email_match:
            patient_info['email'] = email_match.group(1)
            logger.info(f"Extracted email from query: {patient_info['email']}")
        
        # Common patterns for patient names
        # Look for patterns like "ram's height", "height of ram", "ram height", etc.
        name_patterns = [
            r"(\w+)\s*(?:'s|')\s*(?:height|weight|age|profile|data)",
            r"(?:height|weight|age|profile|data)\s+of\s+(\w+)",
            r"(\w+)\s+(?:height|weight|age|profile|data)",
            r"patient\s+(\w+)",
            r"(\w+)\s+patient"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                name = match.group(1)
                # Skip common words that aren't names
                if name not in ['what', 'the', 'is', 'my', 'your', 'their', 'his', 'her', 'patient', 'tell', 'show', 'give']:
                    patient_info['name'] = name
                    logger.info(f"Extracted name from query: {name}")
                    break
        
        return patient_info
    
    def _get_relevant_context(self, query: str, limit: int = 5, patient_email: str = None) -> List[Dict]:
        """Retrieve relevant context from MongoDB based on user query - uses latest document from users collection"""
        latest_doc = None
        try:
            query_lower = query.lower()
            
            # Always get the latest document from users collection (most recent by updatedAt)
            # This ensures we always have the most up-to-date patient data
            try:
                latest_doc = self.data_retrieval.collection.find_one(
                    {},
                    sort=[("updatedAt", -1)]
                )
                if not latest_doc:
                    # Fallback to createdAt if updatedAt doesn't exist
                    latest_doc = self.data_retrieval.collection.find_one(
                        {},
                        sort=[("createdAt", -1)]
                    )
                
                if latest_doc:
                    patient_email_from_doc = latest_doc.get('email', '')
                    logger.info(f"Using latest document from users collection with email: {patient_email_from_doc}, bpReadings count: {len(latest_doc.get('bpReadings', []))}")
                    patient_info = {'email': patient_email_from_doc}
                else:
                    logger.warning("No documents found in users collection")
                    patient_info = {}
            except Exception as e:
                logger.error(f"Error retrieving latest document: {e}")
                patient_info = {}
            
            # Always include the latest document in context so the model has full patient data
            context = []
            if latest_doc:
                bp_readings = latest_doc.get("bpReadings", [])
                profile = latest_doc.get("profile", {})
                context.append({
                    "content": f"Complete Patient Record: Email {latest_doc.get('email')}, bpReadings count: {len(bp_readings)}",
                    "data_type": "patient_profile",
                    "timestamp": None,
                    "metadata": {
                        "email": latest_doc.get("email"),
                        "role": latest_doc.get("role"),
                        "height": profile.get("height") if isinstance(profile, dict) else None,
                        "weight": profile.get("weight") if isinstance(profile, dict) else None,
                        "age": profile.get("age") if isinstance(profile, dict) else None,
                    },
                    "full_document": latest_doc  # Include FULL document for model - this is the complete patient data
                })
                logger.info(f"Added latest document to context: email={latest_doc.get('email')}, bpReadings={len(bp_readings)}")
            
            # If asking about profile data, get documents with profile info
            # Use natural language parser to detect profile queries
            profile_field = extract_profile_field(query)
            is_profile_query = profile_field is not None
            
            # Check for "last N" or "recent N" queries for health metrics (e.g., "last 7 systolic")
            import re
            # More flexible pattern: "last 7", "last 7 days", "last 7 readings", "last 7 systolic", etc.
            # Also matches "last 7 systolic" where systolic comes after the number
            last_n_pattern = r'\b(?:last|recent|past)\s+(\d+)(?:\s+(?:days?|readings?|records?|measurements?))?\b'
            last_n_match = re.search(last_n_pattern, query_lower)
            metric = extract_health_metric(query)
            
            # Log for debugging
            logger.info(f"Query: '{query}', Email: {patient_info.get('email')}, Last N match: {last_n_match.group(1) if last_n_match else None}, Metric: {metric}")
            
            # If asking for "last N" readings of a specific metric, extract directly from bpReadings
            if last_n_match and metric:
                try:
                    n_count = int(last_n_match.group(1))
                    logger.info(f"Processing 'last {n_count} {metric}' query")
                    
                    # Use the latest document we already retrieved
                    if latest_doc and 'bpReadings' in latest_doc:
                        patient_doc = latest_doc
                        bp_readings = patient_doc.get('bpReadings', [])
                        logger.info(f"Found {len(bp_readings)} bpReadings in document for email {patient_info['email']}")
                        # Sort by date descending (most recent first), then by time if available
                        sorted_readings = sorted(
                            bp_readings,
                            key=lambda x: (
                                x.get('date', ''),
                                x.get('time', '')
                            ),
                            reverse=True
                        )
                        
                        # Get the last N readings that have the requested metric
                        metric_field_map = {
                            'systolic': 'systolic',
                            'diastolic': 'diastolic',
                            'pulse': 'pulse',
                            'bmi': 'bmi',
                            'blood sugar': 'fastingBloodSugar',
                            'bloodsugar': 'fastingBloodSugar',
                            'glucose': 'fastingBloodSugar',
                            'cholesterol': 'totalCholesterol',
                            'waist': 'waistCircumference',
                            'sleep': 'sleepQuality',
                            'stress': 'stressLevel'
                        }
                        metric_field = metric_field_map.get(metric.lower())
                        logger.info(f"Metric field mapped: {metric} -> {metric_field}")
                        
                        if metric_field:
                            # Filter readings that have this metric
                            filtered_readings = [
                                r for r in sorted_readings 
                                if r.get(metric_field) is not None
                            ]
                            logger.info(f"Found {len(filtered_readings)} readings with {metric_field} field")
                            # Take the last N
                            selected_readings = filtered_readings[:n_count]
                            logger.info(f"Selected {len(selected_readings)} readings for response")
                            
                            if selected_readings:
                                # Append to existing context (which already has the full document)
                                reading_context = []
                                for reading in selected_readings:
                                    content_parts = []
                                    if reading.get("date"):
                                        content_parts.append(f"Date: {reading['date']}")
                                    if reading.get("time"):
                                        content_parts.append(f"Time: {reading['time']}")
                                    if reading.get(metric_field) is not None:
                                        content_parts.append(f"{metric.capitalize()}: {reading[metric_field]}")
                                    
                                    reading_context.append({
                                        "content": "; ".join(content_parts),
                                        "data_type": "blood_pressure_reading",
                                        "timestamp": reading.get("date"),
                                        "metadata": reading,
                                        "full_document": patient_doc  # Include full document for stats calculation
                                    })
                                
                                # Append reading context to main context
                                context.extend(reading_context)
                                logger.info(f"Found {len(reading_context)} readings for last {n_count} {metric} values")
                                return context
                            else:
                                logger.warning(f"No readings found with {metric_field} field for email {patient_info['email']}")
                        else:
                            logger.warning(f"Could not map metric '{metric}' to database field")
                    else:
                        logger.warning(f"Document found for email {patient_info['email']} but no bpReadings array")
                except Exception as e:
                    logger.error(f"Error processing 'last N' query: {e}", exc_info=True)
            
            # Handle simple metric queries (e.g., "systolic" without "last N")
            # If user asks for a metric but no "last N" or date, get all readings for that metric
            if metric and not last_n_match and latest_doc:
                try:
                    # Use the latest document we already retrieved
                    patient_doc = latest_doc
                    
                    if patient_doc and 'bpReadings' in patient_doc:
                        bp_readings = patient_doc.get('bpReadings', [])
                        # Sort by date descending (most recent first)
                        sorted_readings = sorted(
                            bp_readings,
                            key=lambda x: (
                                x.get('date', ''),
                                x.get('time', '')
                            ),
                            reverse=True
                        )
                        
                        # Get readings that have the requested metric
                        metric_field_map = {
                            'systolic': 'systolic',
                            'diastolic': 'diastolic',
                            'pulse': 'pulse',
                            'bmi': 'bmi',
                            'blood sugar': 'fastingBloodSugar',
                            'bloodsugar': 'fastingBloodSugar',
                            'glucose': 'fastingBloodSugar',
                            'cholesterol': 'totalCholesterol',
                            'waist': 'waistCircumference',
                            'sleep': 'sleepQuality',
                            'stress': 'stressLevel'
                        }
                        metric_field = metric_field_map.get(metric.lower())
                        
                        if metric_field:
                            # Filter readings that have this metric
                            filtered_readings = [
                                r for r in sorted_readings 
                                if r.get(metric_field) is not None
                            ]
                            
                            if filtered_readings:
                                # Append to existing context
                                reading_context = []
                                # Get all readings (or limit to reasonable number)
                                for reading in filtered_readings[:limit * 2]:  # Get more readings for context
                                    content_parts = []
                                    if reading.get("date"):
                                        content_parts.append(f"Date: {reading['date']}")
                                    if reading.get("time"):
                                        content_parts.append(f"Time: {reading['time']}")
                                    if reading.get(metric_field) is not None:
                                        content_parts.append(f"{metric.capitalize()}: {reading[metric_field]}")
                                    
                                    reading_context.append({
                                        "content": "; ".join(content_parts),
                                        "data_type": "blood_pressure_reading",
                                        "timestamp": reading.get("date"),
                                        "metadata": reading,
                                        "full_document": patient_doc  # Include full document for stats calculation
                                    })
                                
                                # Append reading context to main context
                                context.extend(reading_context)
                                logger.info(f"Found {len(reading_context)} readings for {metric}")
                                return context
                except Exception as e:
                    logger.warning(f"Error processing metric query: {e}")
            
            # Try specialized date-based query (for health data)
            # Extract date in any format (natural language or structured)
            date_str = extract_date_from_query(query)
            
            if date_str and latest_doc:
                query_lower = query.lower()
                
                # Check for health metrics using natural language parser
                metric = extract_health_metric(query)
                
                # Search within the latest document's bpReadings array
                if latest_doc.get('bpReadings'):
                    bp_readings = latest_doc.get('bpReadings', [])
                    # Filter readings matching the date
                    matching_readings = [r for r in bp_readings if r.get('date') == date_str]
                    
                    if matching_readings:
                        # Append to existing context
                        reading_context = []
                        for reading in matching_readings[:limit]:
                            # Build content string from reading data
                            content_parts = []
                            if reading.get("date"):
                                content_parts.append(f"Date: {reading['date']}")
                            if reading.get("time"):
                                content_parts.append(f"Time: {reading['time']}")
                            if reading.get("systolic") is not None:
                                content_parts.append(f"Systolic: {reading['systolic']}")
                            if reading.get("diastolic") is not None:
                                content_parts.append(f"Diastolic: {reading['diastolic']}")
                            if reading.get("pulse") is not None:
                                content_parts.append(f"Pulse: {reading['pulse']}")
                            if reading.get("bmi") is not None:
                                content_parts.append(f"BMI: {reading['bmi']}")
                            if reading.get("fastingBloodSugar") is not None:
                                content_parts.append(f"Fasting Blood Sugar: {reading['fastingBloodSugar']}")
                            if reading.get("totalCholesterol") is not None:
                                content_parts.append(f"Total Cholesterol: {reading['totalCholesterol']}")
                            if reading.get("waistCircumference") is not None:
                                content_parts.append(f"Waist Circumference: {reading['waistCircumference']}")
                            if reading.get("sleepQuality") is not None:
                                content_parts.append(f"Sleep Quality: {reading['sleepQuality']}")
                            if reading.get("stressLevel") is not None:
                                content_parts.append(f"Stress Level: {reading['stressLevel']}")
                            
                            reading_context.append({
                                "content": "; ".join(content_parts),
                                "data_type": "blood_pressure_reading",
                                "timestamp": reading.get("date"),
                                "metadata": reading,
                                "full_document": latest_doc  # Include full document for model
                            })
                        
                        if reading_context:
                            # Append reading context to main context
                            context.extend(reading_context)
                            logger.info(f"Found {len(reading_context)} readings for date {date_str}")
                            return context
            
            # Handle profile queries (height, weight, age, etc.)
            if is_profile_query:
                # Use the latest document we already retrieved
                if latest_doc and latest_doc.get('profile'):
                    profile_results = [latest_doc]
                else:
                    # Fallback: try to find any document with profile
                    try:
                        profile_results = list(self.data_retrieval.collection.find(
                            {"profile": {"$exists": True}}
                        ).sort("updatedAt", -1).limit(1))
                    except Exception as e:
                        logger.warning(f"Error retrieving profile: {e}")
                        profile_results = []
                
                if profile_results:
                    # Get the MOST RECENT document (first one after sorting)
                    doc = profile_results[0]
                    # Note: We already have latest_doc in context, but if this is a profile query,
                    # we might want to add more specific profile context
                    profile = doc.get('profile', {})
                    
                    if profile and isinstance(profile, dict):
                        # Extract ALL profile data - preserve complete structure
                        profile_data = {
                            "email": doc.get("email"),
                            "role": doc.get("role"),
                            "clerkId": doc.get("clerkId"),
                            "height": profile.get("height"),
                            "weight": profile.get("weight"),
                            "age": profile.get("age"),
                            "gender": profile.get("gender"),
                            "smoker": profile.get("smoker"),
                            "diabetes": profile.get("diabetes"),
                            "hypertension_treated": profile.get("hypertension_treated"),
                            "family_history_of_cardiovascular_disease": profile.get("family_history_of_cardiovascular_disease"),
                            "atrial_fibrillation": profile.get("atrial_fibrillation"),
                            "chronic_kidney_disease": profile.get("chronic_kidney_disease"),
                            "rheumatoid_arthritis": profile.get("rheumatoid_arthritis"),
                            "chronic_obstructive_pulmonary_disorder": profile.get("chronic_obstructive_pulmonary_disorder"),
                            "profile_createdAt": profile.get("createdAt"),
                            "profile_updatedAt": profile.get("updatedAt")
                        }
                        
                        # Also include ALL bpReadings for context
                        bp_readings = doc.get("bpReadings", [])
                        
                        logger.info(f"Extracted profile data from MOST RECENT document: height={profile_data.get('height')}, weight={profile_data.get('weight')}, age={profile_data.get('age')}, email={profile_data.get('email')}, bpReadings count={len(bp_readings)}")
                        
                        # Extract patient name from email (part before @) for display
                        email_parts = profile_data.get('email', '').split('@')
                        patient_name = email_parts[0] if email_parts else 'Patient'
                        
                        context.append({
                            "content": f"Complete Patient Record: Patient Name: {patient_name}, Email: {profile_data.get('email')}, Height: {profile_data.get('height')} cm, Weight: {profile_data.get('weight')} kg, Age: {profile_data.get('age')} years. Total bpReadings: {len(bp_readings)}",
                            "data_type": "patient_profile",
                            "timestamp": None,
                            "metadata": {**profile_data, "patient_name": patient_name},
                            "full_document": doc,  # Include full document for model
                            "patient_identifier": patient_info  # Store what was requested
                        })
                    
                    if context:
                        logger.info(f"Found profile document with email: {profile_data.get('email')}, patient_name: {context[0].get('metadata', {}).get('patient_name')}")
                        return context
            
            # Build base filter query with email if provided
            base_filter = {}
            if patient_info.get('email'):
                base_filter["email"] = patient_info['email']
                logger.info(f"Filtering all queries by email: {patient_info['email']}")
            
            # Try to search using data_retrieval (works with existing collections)
            # Note: search_data may not support email filter, so we'll filter results after if needed
            search_results = self.data_retrieval.search_data(query, limit=limit)
            
            # Filter search results by email if provided
            if patient_info.get('email') and search_results:
                search_results = [doc for doc in search_results if doc.get('email') == patient_info['email']]
            
            # If no results, try with data_ingestion
            if not search_results:
                try:
                    search_results = self.data_ingestion.search_data(query, limit=limit)
                    # Filter by email if provided
                    if patient_info.get('email') and search_results:
                        search_results = [doc for doc in search_results if doc.get('email') == patient_info['email']]
                except:
                    pass
            
            # Also get latest data if no search results, with email filter
            if not search_results:
                search_results = self.data_retrieval.get_all_data(limit=limit, filter_query=base_filter if base_filter else None)
            
            # If no specific results, get the most recent document to give full context
            if not search_results and not is_profile_query:
                # Get most recent document (sorted by updatedAt), filtered by email if provided
                try:
                    search_results = list(self.data_retrieval.collection.find(base_filter).sort("updatedAt", -1).limit(1))
                    if not search_results:
                        search_results = list(self.data_retrieval.collection.find(base_filter).sort("createdAt", -1).limit(1))
                except:
                    search_results = self.data_retrieval.get_all_data(limit=1, filter_query=base_filter if base_filter else None)
            
            # Format context - handle different document structures
            context = []
            for doc in search_results:
                # Handle profile data if present - include full document
                if "profile" in doc and isinstance(doc["profile"], dict):
                    profile = doc["profile"]
                    
                    # Extract ALL profile data - preserve complete structure
                    profile_data = {
                        "email": doc.get("email"),
                        "role": doc.get("role"),
                        "clerkId": doc.get("clerkId"),
                        "height": profile.get("height"),
                        "weight": profile.get("weight"),
                        "age": profile.get("age"),
                        "gender": profile.get("gender"),
                        "smoker": profile.get("smoker"),
                        "diabetes": profile.get("diabetes"),
                        "hypertension_treated": profile.get("hypertension_treated"),
                        "family_history_of_cardiovascular_disease": profile.get("family_history_of_cardiovascular_disease"),
                        "atrial_fibrillation": profile.get("atrial_fibrillation"),
                        "chronic_kidney_disease": profile.get("chronic_kidney_disease"),
                        "rheumatoid_arthritis": profile.get("rheumatoid_arthritis"),
                        "chronic_obstructive_pulmonary_disorder": profile.get("chronic_obstructive_pulmonary_disorder"),
                        "profile_createdAt": profile.get("createdAt"),
                        "profile_updatedAt": profile.get("updatedAt")
                    }
                    
                    bp_readings = doc.get("bpReadings", [])
                    logger.info(f"Extracted profile from document: height={profile_data.get('height')}, weight={profile_data.get('weight')}, age={profile_data.get('age')}, bpReadings count={len(bp_readings)}")
                    
                    context.append({
                        "content": f"Complete Patient Record: Email {profile_data.get('email')}, Height {profile_data.get('height')} cm, Weight {profile_data.get('weight')} kg, Age {profile_data.get('age')} years. Total bpReadings: {len(bp_readings)}",
                        "data_type": "patient_profile",
                        "timestamp": None,
                        "metadata": profile_data,
                        "full_document": doc  # Include FULL document for model
                    })
                
                # Handle nested bpReadings array
                if "bpReadings" in doc and isinstance(doc["bpReadings"], list):
                    # Only extract bpReadings if query is about readings, not profile
                    if not is_profile_query:
                        # Extract all bpReadings
                        for reading in doc["bpReadings"]:
                            content_parts = []
                            if reading.get("date"):
                                content_parts.append(f"Date: {reading['date']}")
                            if reading.get("time"):
                                content_parts.append(f"Time: {reading['time']}")
                            if reading.get("systolic") is not None:
                                content_parts.append(f"Systolic: {reading['systolic']}")
                            if reading.get("diastolic") is not None:
                                content_parts.append(f"Diastolic: {reading['diastolic']}")
                            if reading.get("pulse") is not None:
                                content_parts.append(f"Pulse: {reading['pulse']}")
                            
                            context.append({
                                "content": "; ".join(content_parts),
                                "data_type": "blood_pressure_reading",
                                "timestamp": reading.get("date"),
                                "metadata": reading
                            })
                else:
                    # Try to extract content from various possible field names
                    content = (doc.get("content") or 
                              doc.get("text") or 
                              doc.get("description") or 
                              doc.get("body") or
                              str(doc.get("_id", "")))
                    
                    # Try to extract data type/type
                    data_type = (doc.get("data_type") or 
                               doc.get("type") or 
                               doc.get("category") or 
                               "document")
                    
                    # Try to extract timestamp
                    timestamp = (doc.get("timestamp") or 
                               doc.get("created_at") or 
                               doc.get("date") or 
                               doc.get("updated_at"))
                    
                    context.append({
                        "content": content,
                        "data_type": data_type,
                        "timestamp": timestamp,
                        "metadata": {k: v for k, v in doc.items() 
                                   if k not in ["content", "text", "description", "body", 
                                              "data_type", "type", "timestamp", "created_at", 
                                              "date", "updated_at", "_id", "bpReadings"]}
                    })
            
            # Latest document is already included in context at the beginning
            # Return context (limit is applied in _format_context_for_prompt if needed)
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _format_context_for_prompt(self, context: List[Dict]) -> str:
        """Format context data for the prompt with complete MongoDB document structure"""
        if not context:
            return "No relevant data found in the database."
        
        formatted = "\n=== COMPLETE PATIENT HEALTH DATA FROM MONGODB ===\n\n"
        
        for i, item in enumerate(context, 1):
            metadata = item.get('metadata', {})
            data_type = item.get('data_type', 'Unknown')
            full_doc = item.get('full_document')  # Get full document if available
            
            # Format patient profile with complete structure
            if data_type == 'patient_profile' or 'height' in str(metadata).lower() or 'weight' in str(metadata).lower() or 'age' in str(metadata).lower():
                formatted += f"--- PATIENT RECORD {i} ---\n"
                
                # Patient basic info
                if metadata.get('email'):
                    formatted += f"Email: {metadata['email']}\n"
                if metadata.get('role'):
                    formatted += f"Role: {metadata['role']}\n"
                if metadata.get('clerkId'):
                    formatted += f"ClerkId: {metadata['clerkId']}\n"
                
                formatted += "\n--- PROFILE OBJECT ---\n"
                
                # Profile metrics - show ALL available
                if metadata.get('height') is not None:
                    formatted += f"Height: {metadata['height']} cm\n"
                if metadata.get('weight') is not None:
                    formatted += f"Weight: {metadata['weight']} kg\n"
                if metadata.get('age') is not None:
                    formatted += f"Age: {metadata['age']} years\n"
                if metadata.get('gender'):
                    formatted += f"Gender: {metadata['gender']}\n"
                if metadata.get('smoker'):
                    formatted += f"Smoker: {metadata['smoker']}\n"
                if metadata.get('diabetes'):
                    formatted += f"Diabetes: {metadata['diabetes']}\n"
                if metadata.get('hypertension_treated'):
                    formatted += f"Hypertension Treated: {metadata['hypertension_treated']}\n"
                if metadata.get('family_history_of_cardiovascular_disease'):
                    formatted += f"Family History of Cardiovascular Disease: {metadata['family_history_of_cardiovascular_disease']}\n"
                if metadata.get('atrial_fibrillation'):
                    formatted += f"Atrial Fibrillation: {metadata['atrial_fibrillation']}\n"
                if metadata.get('chronic_kidney_disease'):
                    formatted += f"Chronic Kidney Disease: {metadata['chronic_kidney_disease']}\n"
                if metadata.get('rheumatoid_arthritis'):
                    formatted += f"Rheumatoid Arthritis: {metadata['rheumatoid_arthritis']}\n"
                if metadata.get('chronic_obstructive_pulmonary_disorder'):
                    formatted += f"Chronic Obstructive Pulmonary Disorder: {metadata['chronic_obstructive_pulmonary_disorder']}\n"
                
                # Include ALL bpReadings if full document is available
                if full_doc and 'bpReadings' in full_doc:
                    bp_readings = full_doc.get('bpReadings', [])
                    formatted += f"\n--- bpReadings Array (Total: {len(bp_readings)} readings) ---\n"
                    
                    for idx, reading in enumerate(bp_readings):
                        formatted += f"\n[Reading {idx}]\n"
                        if reading.get('date'):
                            formatted += f"  Date: {reading['date']}\n"
                        if reading.get('time'):
                            formatted += f"  Time: {reading['time']}\n"
                        if reading.get('systolic') is not None:
                            formatted += f"  Systolic: {reading['systolic']} mmHg\n"
                        if reading.get('diastolic') is not None:
                            formatted += f"  Diastolic: {reading['diastolic']} mmHg\n"
                        if reading.get('pulse') is not None:
                            formatted += f"  Pulse: {reading['pulse']} bpm\n"
                        if reading.get('bmi') is not None:
                            formatted += f"  BMI: {reading['bmi']}\n"
                        if reading.get('fastingBloodSugar') is not None:
                            formatted += f"  Fasting Blood Sugar: {reading['fastingBloodSugar']} mg/dL\n"
                        if reading.get('totalCholesterol') is not None:
                            formatted += f"  Total Cholesterol: {reading['totalCholesterol']} mg/dL\n"
                        if reading.get('waistCircumference') is not None:
                            formatted += f"  Waist Circumference: {reading['waistCircumference']} cm\n"
                        if reading.get('sleepQuality') is not None:
                            formatted += f"  Sleep Quality: {reading['sleepQuality']}/5\n"
                        if reading.get('stressLevel') is not None:
                            formatted += f"  Stress Level: {reading['stressLevel']}/5\n"
                        if reading.get('notes'):
                            formatted += f"  Notes: {reading['notes']}\n"
                        if reading.get('createdAt'):
                            formatted += f"  CreatedAt: {reading['createdAt']}\n"
                
                formatted += "\n"
            
            # Format blood pressure readings
            elif data_type == 'blood_pressure_reading' or 'systolic' in str(metadata).lower():
                formatted += f"--- READING {i} ---\n"
                
                # Date and time
                if metadata.get('date'):
                    formatted += f"📅 Date: {metadata['date']}\n"
                if metadata.get('time'):
                    formatted += f"🕐 Time: {metadata['time']}\n"
                
                formatted += "\n📊 Health Metrics:\n"
                
                # Blood pressure
                if metadata.get('systolic') is not None:
                    formatted += f"  • Systolic Blood Pressure: {metadata['systolic']} mmHg\n"
                if metadata.get('diastolic') is not None:
                    formatted += f"  • Diastolic Blood Pressure: {metadata['diastolic']} mmHg\n"
                if metadata.get('pulse') is not None:
                    formatted += f"  • Pulse Rate: {metadata['pulse']} bpm\n"
                
                # Other metrics
                if metadata.get('bmi') is not None:
                    formatted += f"  • BMI: {metadata['bmi']}\n"
                if metadata.get('fastingBloodSugar') is not None:
                    formatted += f"  • Fasting Blood Sugar: {metadata['fastingBloodSugar']} mg/dL\n"
                if metadata.get('totalCholesterol') is not None:
                    formatted += f"  • Total Cholesterol: {metadata['totalCholesterol']} mg/dL\n"
                if metadata.get('waistCircumference') is not None:
                    formatted += f"  • Waist Circumference: {metadata['waistCircumference']} cm\n"
                if metadata.get('sleepQuality') is not None:
                    formatted += f"  • Sleep Quality: {metadata['sleepQuality']}/5\n"
                if metadata.get('stressLevel') is not None:
                    formatted += f"  • Stress Level: {metadata['stressLevel']}/5\n"
                
                # Notes
                if metadata.get('notes'):
                    formatted += f"\n📝 Notes: {metadata['notes']}\n"
                
                formatted += "\n"
            elif data_type == 'web':
                formatted += f"--- WEB SOURCE {i} ---\n"
                title = metadata.get('title') or 'Untitled'
                url = metadata.get('url') or ''
                snippet = item.get('content', '')
                if len(snippet) > 600:
                    snippet = snippet[:600] + '...'
                formatted += f"Title: {title}\nURL: {url}\nSummary: {snippet}\n\n"
            else:
                # Generic document format
                formatted += f"--- DOCUMENT {i} ---\n"
                content = item.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                formatted += f"Content: {content}\n"
                
                timestamp = item.get('timestamp')
                if timestamp:
                    formatted += f"Date: {timestamp}\n"
                
                formatted += "\n"
        
        formatted += "\n=== END OF DATABASE CONTEXT ===\n"
        return formatted

    # -------------------- Web Research --------------------
    def _web_research(self, query: str) -> List[Dict]:
        """Fetch a few authoritative snippets from the web (Wikipedia, arXiv, Semantic Scholar)."""
        if not REQUESTS_AVAILABLE or not Config.ALLOW_WEB:
            return []
        results: List[Dict] = []
        try:
            # Wikipedia: quick summary via REST and opensearch
            search = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"}, timeout=8
            ).json()
            if search and len(search) >= 4 and search[1]:
                title = search[1][0]
                page = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", timeout=8
                ).json()
                summary = page.get("extract") or ""
                url = page.get("content_urls", {}).get("desktop", {}).get("page")
                if summary:
                    results.append({
                        "data_type": "web",
                        "content": summary,
                        "metadata": {"title": page.get("title", title), "url": url or f"https://en.wikipedia.org/wiki/{title}"}
                    })
        except Exception:
            pass
        try:
            # arXiv API (Atom) - simple query
            import xml.etree.ElementTree as ET
            feed = requests.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "start": 0, "max_results": 1}, timeout=8
            )
            if feed.status_code == 200:
                root = ET.fromstring(feed.text)
                ns = {"a": "http://www.w3.org/2005/Atom"}
                entry = root.find("a:entry", ns)
                if entry is not None:
                    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
                    summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
                    link_el = entry.find("a:link[@type='text/html']", ns)
                    link = link_el.attrib.get("href") if link_el is not None else None
                    if summary:
                        results.append({
                            "data_type": "web",
                            "content": summary,
                            "metadata": {"title": title or "arXiv", "url": link or "https://arxiv.org"}
                        })
        except Exception:
            pass
        try:
            # Semantic Scholar search
            s2 = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 1, "fields": "title,abstract,url,year"}, timeout=8
            ).json()
            if isinstance(s2, dict) and s2.get("data"):
                paper = s2["data"][0]
                abstract = (paper.get("abstract") or "").strip()
                title = paper.get("title")
                url = paper.get("url")
                if abstract:
                    results.append({
                        "data_type": "web",
                        "content": abstract,
                        "metadata": {"title": title or "Semantic Scholar", "url": url}
                    })
        except Exception:
            pass
        return results
    
    def _get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for the session"""
        try:
            history = self.chat_history_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit)
            
            # Reverse to get chronological order
            return list(reversed(list(history)))
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def _save_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Save a message to chat history"""
        try:
            message = {
                "session_id": session_id,
                "role": role,  # 'user' or 'assistant'
                "content": content,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }
            self.chat_history_collection.insert_one(message)
        except Exception as e:
            logger.error(f"Error saving message: {e}")
    
    def _call_openrouter_api(self, messages: List[Dict]) -> str:
        """Call OpenRouter API with messages"""
        try:
            url = f"{self.openrouter_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # Optional: for tracking
                "X-Title": "MongoDB Chatbot"  # Optional: for tracking
            }
            
            payload = {
                "model": self.openrouter_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000  # Increased for comprehensive recommendations
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"OpenRouter error details: {error_data}")
                except:
                    logger.error(f"OpenRouter error response: {e.response.text}")
            raise
    
    def generate_response(self, user_query: str, session_id: str = "default", patient_email: str = None) -> str:
        """
        Generate chatbot response using RAG from MongoDB data
        
        Args:
            user_query: The user's question/query
            session_id: Session identifier for conversation history
            patient_email: Email of the logged-in patient to filter data
            
        Returns:
            The chatbot's response
        """
        try:
            # Save user message
            self._save_message(session_id, "user", user_query)
            
            # Get relevant context from MongoDB, filtered by patient email if provided
            context = self._get_relevant_context(user_query, patient_email=patient_email)
            # If database context is empty or clearly out-of-scope, augment with web research
            if (not context or len(context) == 0) and Config.ALLOW_WEB:
                web_ctx = self._web_research(user_query)
                context = web_ctx
            else:
                # Also consider augmenting DB context with 1-2 web snippets for broader answers
                if Config.ALLOW_WEB and len(context) < 2:
                    web_ctx = self._web_research(user_query)
                    context = context + web_ctx[:1]
            formatted_context = self._format_context_for_prompt(context)
            
            # Get conversation history
            history = self._get_conversation_history(session_id)
            
            # Build comprehensive system prompt
            system_prompt = """You are a medical health data assistant chatbot that answers questions for PATIENTS based on their health data stored in MongoDB Atlas.

IMPORTANT: Always address the patient directly using "your" or "patient's" language. You are talking TO the patient, not ABOUT the patient.

DATA STRUCTURE UNDERSTANDING:
The database contains patient health records stored as MongoDB documents. Each document has the EXACT structure shown below:

**COMPLETE MONGODB DOCUMENT STRUCTURE:**
{
  "_id": "document_id",
  "clerkId": "user_id",
  "__v": 0,
  "createdAt": "timestamp",
  "email": "patient_email@example.com",
  "role": "patient",
  "updatedAt": "timestamp",
  "bpReadings": [
    {
      "date": "2025-11-05",
      "time": "12:21",
      "systolic": 118,
      "diastolic": 76,
      "pulse": 72,
      "bmi": 23.4,
      "fastingBloodSugar": 92,
      "totalCholesterol": 165,
      "waistCircumference": 82,
      "sleepQuality": 4,
      "stressLevel": 2,
      "notes": "Felt relaxed, normal morning routine",
      "createdAt": "timestamp",
      "_id": "reading_id"
    },
    // ... more readings can be added to this array
  ],
  "profile": {
    "gender": "male",
    "age": 20,
    "height": 183,
    "weight": 72,
    "smoker": "yes",
    "hypertension_treated": "no",
    "family_history_of_cardiovascular_disease": "yes",
    "atrial_fibrillation": "yes",
    "chronic_kidney_disease": "yes",
    "rheumatoid_arthritis": "yes",
    "diabetes": "yes",
    "chronic_obstructive_pulmonary_disorder": "yes",
    "createdAt": "timestamp",
    "updatedAt": "timestamp",
    "_id": "profile_id"
  }
}

IMPORTANT:
- Each MongoDB document represents ONE patient record
- The "profile" object contains patient profile information (height, weight, age, etc.)
- The "bpReadings" array contains ALL health readings for that patient
- When new readings are added, they are appended to the bpReadings array
- The profile object contains the CURRENT patient information
- Use the EXACT values from the profile object when answering profile questions (e.g., height, weight, age)
- If there are multiple documents in the database, the system will retrieve the MOST RECENT document (sorted by updatedAt)
- Always use the values from the MOST RECENT document's profile object
- The context provided contains the COMPLETE document structure with ALL fields

INSTRUCTIONS FOR ANSWERING QUESTIONS:

CRITICAL: Answer ONLY what was asked. Do NOT dump all data. Be PRECISE, CONCISE, and SUMMARIZED.

**CONCISENESS REQUIREMENT:**
- Keep each bullet point to ONE SHORT SENTENCE (ideally 10-15 words maximum)
- Summarize information - do NOT provide lengthy explanations
- Focus on KEY points only - skip unnecessary details
- For diet recommendations: List specific actions briefly (e.g., "Limit sodium to <1500mg/day" not "Aim to limit your sodium intake to less than 1,500 mg per day to help manage your blood pressure")
- Avoid repetition and verbose explanations
- Be direct and to-the-point

1. **Simple Profile Queries:**
   - If a SPECIFIC patient name or email is mentioned in the query:
     * Use the patient's name: "[Patient Name]'s height is [value] cm"
     * Example: "what is ram's height" → Answer: "Ram's height is [value] cm"
     * Example: "ram height" → Answer: "Ram's height is [value] cm"
     * If name is in email format, extract the name part (before @)
   - If NO specific patient is mentioned (general query like "what is height"):
     * Address the patient directly: "Your height is [value] cm"
     * Use "your" when referring to the current/queried patient's data
   - If asked "what is height" (no name) → Answer: "Your height is [value] cm" (ONLY this, nothing else)
   - If asked "what is ram's height" → Answer: "Ram's height is [value] cm"
   - If asked "what is weight" → Answer: "Your weight is [value] kg" (ONLY this)
   - If asked "what is age" → Answer: "Your age is [value] years" (ONLY this)
   - If asked "how tall am i" → Answer: "Your height is [value] cm"
   - If asked "how much do i weigh" → Answer: "Your weight is [value] kg"
   - If asked "how old am i" → Answer: "Your age is [value] years"
   - Extract ONLY the requested field from the profile object
   - Do NOT show all profile data or bpReadings

2. **Date-Based Queries:**
   - Users can ask about dates in ANY format:
     * Structured: "2025-11-01", "2025-11-05"
     * Natural language: "1 november 2025", "1st november 2025", "november 1 2025"
     * Other formats: "11/1/2025", "1-11-2025"
   - The system automatically converts all date formats to YYYY-MM-DD for database lookup
   - When user asks about a specific date, search the bpReadings array for entries matching that date
   - Extract ONLY the exact value requested
   - Format your answer clearly using patient language: "On [date] at [time], your [metric] was [value] [unit]"
   - Examples:
     * "On November 1, 2025 at 02:00, your systolic blood pressure was 128 mmHg"
     * "Your heart rate on November 1, 2025 was 72 bpm"
     * "On 2025-11-01, your blood sugar was 92 mg/dL"
   - Always use "your" when referring to the patient's health metrics
   - Always show the date in a readable format in your response when possible
   - Include notes if available
   - Do NOT show all readings or all metrics

3. **Health Metric Synonyms:**
   Users can ask about metrics using various terms:
   - Systolic: "systolic", "systolic blood pressure", "upper bp", "top number", "first number"
   - Diastolic: "diastolic", "diastolic blood pressure", "lower bp", "bottom number", "second number"
   - Pulse: "pulse", "heart rate", "heartbeat", "bpm", "hr"
   - Blood Sugar: "blood sugar", "glucose", "blood glucose", "fasting blood sugar", "sugar level"
   - Cholesterol: "cholesterol", "total cholesterol", "cholesterol level"
   - Waist: "waist", "waist circumference", "waist size", "waist measurement"
   - Sleep: "sleep", "sleep quality", "sleep score", "how i slept"
   - Stress: "stress", "stress level", "stress score", "how stressed", "anxiety level"
   - BMI: "bmi", "body mass index", "body mass"

4. **Metric Values:**
   - Systolic/Diastolic: Always include "mmHg" unit
   - Pulse: Always include "bpm" unit
   - Blood Sugar: Always include "mg/dL" unit
   - Cholesterol: Always include "mg/dL" unit
   - Waist Circumference: Always include "cm" unit
   - Height: Always include "cm" unit
   - Weight: Always include "kg" unit
   - BMI: No unit needed (it's a ratio)
   - Sleep Quality/Stress Level: Include "/5" scale

4. **Trend Analysis:**
   - If asked about trends or comparisons, compare values across different dates
   - Calculate changes/differences when relevant
   - Identify patterns (e.g., "Your systolic has been decreasing over the past week")
   - Only show relevant data, not everything

5. **Profile Information:**
   Users can ask about profile fields using various terms:
   - Height: "height", "how tall", "tall", "height in cm"
   - Weight: "weight", "how much do i weigh", "body weight", "my weight"
   - Age: "age", "how old", "my age", "years old"
   - Gender: "gender", "sex", "male or female"
   - Smoker: "smoker", "smoking", "do i smoke", "smoking status"
   - Diabetes: "diabetes", "have diabetes", "diabetic", "diabetes status"
   - Hypertension: "hypertension", "high blood pressure treated", "bp treated"
   - Family History: "family history", "family history of heart disease", "family heart disease"
   - Atrial Fibrillation: "atrial fibrillation", "afib", "a-fib"
   - Kidney Disease: "kidney disease", "chronic kidney disease", "ckd", "kidney problems"
   - Arthritis: "arthritis", "rheumatoid arthritis", "ra", "joint problems"
   - COPD: "copd", "chronic obstructive pulmonary disorder", "lung disease", "breathing problems"
   - When asked about specific profile field, extract ONLY that field
   - Do NOT show all profile data unless specifically asked
   - Use profile data to provide context only when relevant

6. **Missing Data:**
   - If a specific reading doesn't have a particular metric, clearly state "This metric is not available for this date"
   - If a date doesn't exist, say "No readings found for this date"
   - If profile data is missing, say "Height information is not available in your profile"

7. **Response Format:**
   - CRITICAL: ALL answers MUST be formatted in bullet points
   - MANDATORY LINE BREAK RULE: Whenever you start a new bullet point, you MUST start it on a NEW LINE
   - Each bullet point must be on its own separate line - NEVER put multiple bullets on the same line
   - When you write "• " or "- ", you MUST immediately go to a new line after the bullet symbol
   - Add a blank line after each bullet point (press Enter twice after each bullet)
   - CONCISENESS: Keep each bullet point to ONE SHORT SENTENCE (10-15 words maximum)
   - SUMMARIZE: Provide summarized, to-the-point information - avoid lengthy explanations
   - For recommendations: List actions briefly (e.g., "Limit sodium <1500mg/day" not long explanations)
   - Use markdown bullet format: "• " or "- " at the start of each NEW LINE
   - For simple single-value answers (e.g., "what is height"), you can use one bullet: "• Your height is 183 cm"
   - For complex answers (recommendations, multi-part questions, diet advice), use multiple bullets with each point on its own line and a blank line between bullets
   - NEVER write bullets like this: "• Point 1 • Point 2" - ALWAYS write: "• Point 1\n\n• Point 2"
   - Always address the PATIENT directly using "your" language
   - Use natural, friendly patient language
   - Include units when relevant
   - Do NOT dump all database data
   - Examples:
     * Simple query "what is height" → Answer: "• Your height is 183 cm"
     * Date query "systolic on 1st november 2025" → Answer: "• On November 1, 2025 at 02:00, your systolic blood pressure was 128 mmHg"
     * Diet recommendation query → Answer in multiple bullets (each on separate line with a blank line after each):
       • Your average systolic is 128.71 mmHg

       • Follow DASH diet principles

       • Limit sodium to <1500 mg/day

       • Increase fruits and vegetables (4–5 servings each daily)

       • Choose whole grains over refined grains

       • Include lean proteins (fish, poultry, legumes)

       • Medical disclaimer: This is general guidance. Always consult your doctor before acting.
   - Always frame responses as if talking directly TO the patient

8. **KNOWLEDGE SOURCES AND OUT-OF-SCOPE QUESTIONS:**
   - PRIMARY: Use database context when available for patient-specific data
   - SECONDARY: When database context is insufficient or question is out of scope:
     * Use your training knowledge (Gemini 2.5) based on medical research, clinical guidelines, and evidence-based practices
     * Use web sources provided in context (Wikipedia, arXiv research papers, Semantic Scholar abstracts)
     * Synthesize information from multiple sources to provide comprehensive, evidence-based answers
     * Reference authoritative sources when available (e.g., "According to [source]..." or "Research suggests...")
   
   - For questions about:
     * Diet recommendations → Use evidence from DASH diet studies, nutrition research, dietary guidelines
     * Lifestyle interventions → Reference exercise guidelines, stress management research, sleep hygiene studies
     * Health conditions → Use medical knowledge, clinical guidelines, peer-reviewed research
     * Statistical analysis → Calculate mean, median, mode, min, max, standard deviation from database data
     * General health advice → Combine database insights with evidence-based recommendations from research
   
   - ALWAYS provide actionable, evidence-based guidance when asked
   - DO NOT refuse to answer questions outside database scope - instead use your knowledge and provided web sources
   - When providing recommendations:
     * Base them on the calculated values from database (e.g., if average systolic is 128.71 mmHg, provide guidance appropriate for that range)
     * Reference established medical guidelines (e.g., AHA blood pressure categories, ADA diabetes guidelines)
     * Cite research when available in context (briefly)
     * Provide specific, practical advice in SUMMARIZED form (e.g., "Limit sodium <1500mg/day" not "Aim to limit your sodium intake to less than 1,500 mg per day to help manage your blood pressure")
     * Keep each recommendation to ONE SHORT SENTENCE (10-15 words) - summarize, don't explain in detail

9. **MEDICAL DISCLAIMER:**
   - When providing ANY health, diet, or lifestyle recommendation, you MUST append this exact sentence as the LAST bullet point:
     "• Medical disclaimer: This is general guidance. Always consult your doctor before acting."
   - This disclaimer is REQUIRED for all recommendations, regardless of source (database, web, or your knowledge)
   - Format it as a bullet point on its own line

IMPORTANT: 
- Answer ONLY the specific question asked
- Do NOT show all data or all readings unless specifically requested
- Extract and return ONLY the requested information
- For database queries: Always base answers on data provided in the context
- For out-of-scope questions: Use your knowledge, web sources in context, and evidence-based guidelines
- When providing numerical values, always include appropriate units
- Be precise with dates and times from the database
- For combined questions (e.g., "average systolic last 7 days and what diet should I eat"):
  * FIRST: Calculate the statistic from database data
  * SECOND: Provide evidence-based dietary/lifestyle recommendations appropriate for that value range
  * Use your knowledge of medical guidelines (e.g., AHA BP categories, DASH diet principles) to tailor advice
  * Reference the calculated value in your recommendations (e.g., "Given your average systolic of 128.71 mmHg...")

CRITICAL FORMATTING REQUIREMENT:
- ALL responses MUST be formatted in bullet points
- MANDATORY LINE BREAK RULE: Whenever you start a new bullet point, you MUST start it on a NEW LINE
- Each bullet point must be on its own separate line - NEVER put multiple bullets on the same line
- When you write "• " or "- ", you MUST immediately go to a new line after the bullet symbol
- Add a blank line after each bullet point (press Enter twice after each bullet)
- Use "• " or "- " at the start of each NEW LINE
- CONCISENESS: Keep each bullet to ONE SHORT SENTENCE (10-15 words maximum) - SUMMARIZE, don't explain in detail
- Simple answers: one bullet point
- Complex answers (recommendations, multi-part): multiple bullet points, each on its own line with a blank line between bullets
- NEVER write bullets like this: "• Point 1 • Point 2" - ALWAYS write each bullet on a new line
- For recommendations: List actions briefly (e.g., "Limit sodium <1500mg/day" not long explanations)
- This formatting is MANDATORY for all responses"""
            
            # Build messages for API calls
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history
            for msg in history[-5:]:  # Last 5 messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Add current context and query with detailed instructions
            user_message = f"""=== CONTEXT (DATABASE + OPTIONAL WEB) ===

Below is relevant information from your MongoDB database and, when needed, reputable web sources (Wikipedia/arXiv/Semantic Scholar). Use this to answer the question accurately.

{formatted_context}

=== YOUR QUESTION ===
{user_query}

=== INSTRUCTIONS ===
CRITICAL: Answer ONLY what was asked. Do NOT show all data. Be CONCISE and SUMMARIZED.

**CONCISENESS REQUIREMENT:**
- Keep each bullet point to ONE SHORT SENTENCE (10-15 words maximum)
- Summarize information - avoid lengthy explanations
- Focus on KEY points only - skip unnecessary details
- For diet recommendations: List specific actions briefly (e.g., "Limit sodium <1500mg/day" not long explanations)
- Avoid repetition and verbose language
- Be direct and to-the-point

Please analyze the database context above and answer the user's question PRECISELY and CONCISELY:
1. Check if a SPECIFIC patient name or email was mentioned in the query
2. If a SPECIFIC patient is mentioned (e.g., "ram's height", "what is john's height"):
   - Use the patient's name: "[Patient Name]'s height is [value] cm"
   - Example: "what is ram's height" → Answer: "Ram's height is 183 cm"
   - Example: "ram height" → Answer: "Ram's height is 183 cm"
3. If NO specific patient is mentioned (e.g., "what is height", "what is my height"):
   - Use "your": "Your height is [value] cm"
4. Identify what specific information is being asked (e.g., height, systolic on a date, etc.)
5. Extract ONLY that specific value from the context
6. If asking about profile (height, weight, age): Extract ONLY from profile object, ignore bpReadings
7. If asking about a specific date/metric: Extract ONLY that metric for that date, ignore other dates
8. Include units when relevant (cm, kg, mmHg, etc.)
9. Be concise - one sentence or short answer is usually enough
10. Do NOT dump all database data or all readings
11. If data is not found, clearly state that

Examples:
- User asks "what is height" (no name) → Answer: "Your height is 183 cm"
- User asks "what is ram's height" → Answer: "Ram's height is 183 cm"
- User asks "ram height" → Answer: "Ram's height is 183 cm"
- User asks "systolic on 2025-11-05" → Answer: "On November 5, 2025 at 12:21, your systolic blood pressure was 118 mmHg"
- User asks "what is ram's weight" → Answer: "Ram's weight is 72 kg"
- User asks "what is my weight" → Answer: "Your weight is 72 kg"

CRITICAL: 
- If a patient name is mentioned, use "[Name]'s [field] is [value]"
- If no name is mentioned, use "Your [field] is [value]"
- The context shows which patient's data is being used (check patient_name or email in metadata)

=== CRITICAL INSTRUCTIONS FOR THIS RESPONSE ===

**FORMATTING REQUIREMENT - MANDATORY:**
- ALL answers MUST be formatted in bullet points
- CRITICAL LINE BREAK RULE: Whenever you start a new bullet point, you MUST start it on a NEW LINE
- Each bullet point must be on its own separate line - NEVER put multiple bullets on the same line
- When you write "• " or "- ", you MUST immediately go to a new line after the bullet symbol
- Add a blank line after each bullet point (press Enter twice after each bullet)
- Keep each bullet point concise (ideally ≤ 1 sentence)
- Use "• " or "- " at the start of each NEW LINE
- For simple single-value answers: use one bullet point
- For complex answers (recommendations, multi-part questions): use multiple bullet points, each on its own line with a blank line between bullets
- NEVER write bullets like this: "• Point 1 • Point 2" - ALWAYS write each bullet on a new line: "• Point 1\n\n• Point 2"
- Example format (each bullet on a new line with blank line after):
  • Point 1

  • Point 2

  • Point 3

1. **If the question involves database data (e.g., "average systolic last 7 days"):**
   - Calculate the requested statistic (mean, median, mode, min, max, std) from the bpReadings array in the context
   - Present the calculated value with appropriate units IN BULLET POINT FORMAT
   - CRITICAL: Each bullet point MUST start on a NEW LINE - when you write "• " or "- ", immediately go to a new line

2. **If the question asks for recommendations, advice, or guidance (e.g., "what diet should I eat", "how to lower", "what should I do"):**
   - Use your knowledge (Gemini 2.5) of medical research, clinical guidelines, and evidence-based practices
   - Reference web sources provided in context (Wikipedia, arXiv, Semantic Scholar) when available
   - Base recommendations on the calculated values from database (e.g., if average systolic is 128.71 mmHg, provide guidance for that BP category)
   - Use established medical guidelines (AHA BP categories, DASH diet principles, ADA guidelines, etc.)
   - Provide specific, actionable advice in SUMMARIZED form (e.g., "Limit sodium <1500mg/day" not "Aim to limit your sodium intake to less than 1,500 mg per day to help manage your blood pressure")
   - DO NOT refuse to provide guidance - always give evidence-based recommendations
   - FORMAT ALL RECOMMENDATIONS AS BULLET POINTS, each point on a separate line with a blank line after each
   - CRITICAL: Each bullet point MUST start on a NEW LINE - when you write "• " or "- ", immediately go to a new line
   - CONCISENESS: Keep each recommendation to ONE SHORT SENTENCE (10-15 words) - summarize, don't explain in detail

3. **For combined questions (e.g., "average systolic last 7 days and what diet should I eat"):**
   - FIRST: Calculate and state the statistic from database (in bullet point) - keep it brief
   - SECOND: Provide evidence-based dietary/lifestyle recommendations tailored to that value (each recommendation as a separate bullet point)
   - Reference the calculated value briefly in your recommendations
   - Use your knowledge of medical guidelines to provide appropriate guidance
   - FORMAT ENTIRE RESPONSE IN BULLET POINTS with a blank line after each bullet
   - CRITICAL: Each bullet point MUST start on a NEW LINE - when you write "• " or "- ", immediately go to a new line
   - CONCISENESS: Keep each bullet to ONE SHORT SENTENCE (10-15 words) - summarize recommendations, avoid lengthy explanations

4. **Sources and Citations:**
   - If using web sources from context, mention them briefly in a bullet point (e.g., "• According to research..." or "• Studies suggest...")
   - You can reference your training knowledge without needing to cite every detail
   - Focus on providing helpful, accurate information

5. **Medical Disclaimer:**
   - ALWAYS append this exact sentence as the LAST bullet point if you provide ANY health/diet/lifestyle recommendation:
     • Medical disclaimer: This is general guidance. Always consult your doctor before acting.

**REMEMBER: Format your entire response in bullet points. CRITICAL: Each bullet point MUST start on a NEW LINE. When you write "• " or "- ", you MUST immediately go to a new line after the bullet symbol. Add a blank line after each bullet point. Keep each bullet to ONE SHORT SENTENCE (10-15 words) - SUMMARIZE, don't explain in detail. This is MANDATORY.**
"""
            messages.append({"role": "user", "content": user_message})
            
            # Try OpenRouter first (free models)
            if self.use_openrouter:
                try:
                    assistant_response = self._call_openrouter_api(messages)
                    logger.info("Successfully used OpenRouter API")
                except Exception as openrouter_error:
                    error_msg = str(openrouter_error)
                    logger.warning(f"OpenRouter API error: {error_msg}")
                    # Fall back to OpenAI or fallback mode
                    if self.use_openai:
                        try:
                            # Try OpenAI as fallback
                            response = self.client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1000  # Increased for comprehensive recommendations
                            )
                            assistant_response = response.choices[0].message.content
                            logger.info("Fallback to OpenAI successful")
                        except Exception as openai_error:
                            logger.warning(f"OpenAI also failed: {openai_error}")
                            assistant_response = self._generate_fallback_response(user_query, context, formatted_context)
                    else:
                        assistant_response = self._generate_fallback_response(user_query, context, formatted_context)
            
            # Try OpenAI if OpenRouter not available
            elif self.use_openai:
                try:
                    # Call OpenAI API
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000  # Increased for comprehensive recommendations
                    )
                    
                    assistant_response = response.choices[0].message.content
                except Exception as openai_error:
                    # Handle OpenAI API errors (quota, rate limits, etc.)
                    error_msg = str(openai_error)
                    logger.warning(f"OpenAI API error: {error_msg}")
                    
                    # Check for specific error types
                    if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                        logger.warning("OpenAI quota exceeded. Using fallback mode.")
                        assistant_response = self._generate_fallback_response(user_query, context, formatted_context)
                    elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                        logger.warning("OpenAI rate limit exceeded. Using fallback mode.")
                        assistant_response = self._generate_fallback_response(user_query, context, formatted_context)
                    else:
                        # Other OpenAI errors - use fallback
                        logger.warning(f"OpenAI error: {error_msg}. Using fallback mode.")
                        assistant_response = self._generate_fallback_response(user_query, context, formatted_context)
            else:
                # Fallback: Intelligent response without OpenAI
                assistant_response = self._generate_fallback_response(user_query, context, formatted_context)
            
            # Format bullet points in the response
            assistant_response = self._format_bullet_points(assistant_response)
            
            # Save assistant response
            self._save_message(session_id, "assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Try to provide a helpful error message
            error_str = str(e)
            if "quota" in error_str.lower() or "429" in error_str:
                return ("I encountered an API quota/rate limit issue. The chatbot is now running in fallback mode "
                       "and will use your MongoDB data directly. The responses will still be helpful, but for "
                       "more sophisticated AI responses, please check your OpenAI account billing.")
            return f"I encountered an error: {error_str[:100]}. Please try again or rephrase your question."
    
    def _generate_fallback_response(self, user_query: str, context: List[Dict], formatted_context: str) -> str:
        """
        Generate intelligent response without OpenAI API
        Uses template-based responses with context from MongoDB
        """
        query_lower = user_query.lower()

        # Stats queries (average/mean/median/mode/min/max/std over last N days)
        metric_key, stat_type, days = self._parse_stat_request(user_query)
        if metric_key and stat_type and context:
            latest_doc = None
            for item in context:
                full_doc = item.get('full_document')
                if full_doc and 'bpReadings' in full_doc:
                    latest_doc = full_doc
                    break
            if not latest_doc:
                for item in context:
                    if item.get('data_type') == 'patient_profile' and item.get('full_document'):
                        latest_doc = item['full_document']
                        break
            if latest_doc:
                stats = self._compute_metric_stats(latest_doc.get('bpReadings', []), metric_key, days)
                if stats and stats.get('count'):
                    unit = self._unit_for_metric(metric_key)
                    value = stats.get(stat_type)
                    if value is not None:
                        days_txt = f" over the last {days} days" if days else ""
                        metric_display = {
                            'systolic': 'systolic blood pressure', 'diastolic': 'diastolic blood pressure',
                            'pulse': 'pulse', 'fastingBloodSugar': 'fasting blood sugar',
                            'totalCholesterol': 'total cholesterol', 'waistCircumference': 'waist circumference',
                            'bmi': 'BMI', 'sleepQuality': 'sleep quality', 'stressLevel': 'stress level'
                        }.get(metric_key, metric_key)
                        try:
                            value_str = f"{float(value):.2f}"
                        except Exception:
                            value_str = str(value)
                        response = f"Your {stat_type} {metric_display}{days_txt} is {value_str}{(' ' + unit) if unit else ''}."
                        
                        # If user asks for recommendations, note that AI model should provide them
                        if any(word in query_lower for word in ['diet', 'recommend', 'should', 'what to', 'how to', 'advice', 'guidance']):
                            response += "\n\n[Note: For personalized dietary and lifestyle recommendations based on this value, please ensure the AI model is available. The model will provide evidence-based guidance using medical research and guidelines.]"
                        
                        return response
        
        # Check for profile queries first (height, weight, age, etc.)
        profile_field = extract_profile_field(user_query)
        if profile_field and context:
            # Extract patient identifier to determine response format
            patient_info = self._extract_patient_identifier(user_query)
            patient_name = None
            
            # Find profile data in context
            for item in context:
                metadata = item.get('metadata', {})
                data_type = item.get('data_type', '')
                patient_identifier = item.get('patient_identifier', {})
                
                # If this is profile data, extract the specific field
                if data_type == 'patient_profile' or 'height' in str(metadata).lower() or 'weight' in str(metadata).lower() or 'age' in str(metadata).lower():
                    value = metadata.get(profile_field)
                    
                    # Determine patient name for response
                    if patient_info.get('name'):
                        patient_name = patient_info['name'].title()  # Capitalize first letter
                    elif metadata.get('patient_name'):
                        patient_name = metadata.get('patient_name')
                    elif patient_info.get('email'):
                        # Extract name from email
                        email_parts = patient_info['email'].split('@')
                        patient_name = email_parts[0].title() if email_parts else None
                    elif metadata.get('email'):
                        # Extract name from email in metadata
                        email_parts = metadata.get('email', '').split('@')
                        patient_name = email_parts[0].title() if email_parts else None
                    
                    if value is not None:
                        # Format the response based on whether patient name is specified
                        if patient_name:
                            # Use patient name format
                            if profile_field == 'height':
                                return f"{patient_name}'s height is {value} cm."
                            elif profile_field == 'weight':
                                return f"{patient_name}'s weight is {value} kg."
                            elif profile_field == 'age':
                                return f"{patient_name}'s age is {value} years."
                            elif profile_field == 'gender':
                                return f"{patient_name}'s gender is {value}."
                            elif profile_field == 'smoker':
                                return f"{patient_name}'s smoking status: {value}."
                            elif profile_field == 'diabetes':
                                return f"{patient_name}'s diabetes status: {value}."
                            else:
                                field_display = profile_field.replace('_', ' ').title()
                                return f"{patient_name}'s {field_display.lower()} is {value}."
                        else:
                            # Use "your" format
                            if profile_field == 'height':
                                return f"Your height is {value} cm."
                            elif profile_field == 'weight':
                                return f"Your weight is {value} kg."
                            elif profile_field == 'age':
                                return f"Your age is {value} years."
                            elif profile_field == 'gender':
                                return f"Your gender is {value}."
                            elif profile_field == 'smoker':
                                return f"Smoking status: {value}."
                            elif profile_field == 'diabetes':
                                return f"Diabetes status: {value}."
                            else:
                                field_display = profile_field.replace('_', ' ').title()
                                return f"Your {field_display.lower()} is {value}."
                    else:
                        field_display = profile_field.replace('_', ' ').title()
                        return f"{field_display} information is not available in your profile."
        
        # Check if this is a specific metric query (e.g., "systolic on date 2025-11-05" or "systolic on 1st november 2025")
        date_str = extract_date_from_query(user_query)
        
        if date_str and context:
            
            # Find which metric is being asked using natural language parser
            asked_metric = extract_health_metric(user_query)
            
            # If we found a specific metric, extract and return direct answer
            if asked_metric and context:
                for item in context:
                    metadata = item.get('metadata', {})
                    if metadata.get('date') == date_str:
                        metric_value = metadata.get(asked_metric)
                        if metric_value is not None:
                            # Format the metric name for display
                            metric_display = {
                                'systolic': 'Systolic blood pressure',
                                'diastolic': 'Diastolic blood pressure',
                                'pulse': 'Pulse rate',
                                'bmi': 'BMI',
                                'fastingBloodSugar': 'Fasting blood sugar',
                                'totalCholesterol': 'Total cholesterol',
                                'waistCircumference': 'Waist circumference',
                                'sleepQuality': 'Sleep quality',
                                'stressLevel': 'Stress level'
                            }.get(asked_metric, asked_metric)
                            
                            # Build direct answer in patient language
                            # Format date nicely
                            from datetime import datetime
                            try:
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                formatted_date = date_obj.strftime('%B %d, %Y')  # November 1, 2025
                            except:
                                formatted_date = date_str
                            
                            response = f"On {formatted_date}, your {metric_display.lower()} was {metric_value}."
                            
                            # Add time if available
                            if metadata.get('time'):
                                response = f"On {formatted_date} at {metadata['time']}, your {metric_display.lower()} was {metric_value}."
                            
                            # Add unit information if relevant
                            if asked_metric in ['systolic', 'diastolic']:
                                response += " mmHg"
                            elif asked_metric == 'pulse':
                                response += " bpm"
                            elif asked_metric == 'fastingBloodSugar':
                                response += " mg/dL"
                            elif asked_metric == 'totalCholesterol':
                                response += " mg/dL"
                            elif asked_metric == 'waistCircumference':
                                response += " cm"
                            
                            # Add notes if available
                            if metadata.get('notes'):
                                response += f"\n\nNotes: {metadata['notes']}"
                            
                            return response
            
            # If asking for all data on a date
            if any(word in query_lower for word in ['on date', 'on', 'date']):
                for item in context:
                    metadata = item.get('metadata', {})
                    if metadata.get('date') == date_str:
                        # Return formatted summary of all readings for that date
                        response = f"Here are your health readings for {date_str}:\n\n"
                        
                        if metadata.get('time'):
                            response += f"Time: {metadata['time']}\n"
                        
                        if metadata.get('systolic') is not None:
                            response += f"• Systolic: {metadata['systolic']} mmHg\n"
                        if metadata.get('diastolic') is not None:
                            response += f"• Diastolic: {metadata['diastolic']} mmHg\n"
                        if metadata.get('pulse') is not None:
                            response += f"• Pulse: {metadata['pulse']} bpm\n"
                        if metadata.get('bmi') is not None:
                            response += f"• BMI: {metadata['bmi']}\n"
                        if metadata.get('fastingBloodSugar') is not None:
                            response += f"• Fasting Blood Sugar: {metadata['fastingBloodSugar']} mg/dL\n"
                        if metadata.get('totalCholesterol') is not None:
                            response += f"• Total Cholesterol: {metadata['totalCholesterol']} mg/dL\n"
                        if metadata.get('waistCircumference') is not None:
                            response += f"• Waist Circumference: {metadata['waistCircumference']} cm\n"
                        if metadata.get('sleepQuality') is not None:
                            response += f"• Sleep Quality: {metadata['sleepQuality']}/5\n"
                        if metadata.get('stressLevel') is not None:
                            response += f"• Stress Level: {metadata['stressLevel']}/5\n"
                        
                        if metadata.get('notes'):
                            response += f"\nNotes: {metadata['notes']}"
                        
                        return response
        
        # If we have context, provide intelligent summary
        if context:
            # Extract key information from context
            summary_points = []
            for item in context[:3]:  # Top 3 most relevant
                content = item.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                summary_points.append(content)
            
            # Build response based on query type
            if any(word in query_lower for word in ["what", "tell me", "show me", "explain"]):
                response = f"Based on the data in your MongoDB database:\n\n"
                response += formatted_context
                response += f"\n\nRegarding your question: '{user_query}'\n\n"
                response += "Here's what I found in your database. The information above contains relevant details that should help answer your question."
            
            elif any(word in query_lower for word in ["how many", "count", "total", "number"]):
                response = f"I found {len(context)} relevant document(s) in your database related to '{user_query}'.\n\n"
                response += formatted_context
                response += "\n\nThis shows the relevant information from your database."
            
            elif any(word in query_lower for word in ["latest", "recent", "new", "update"]):
                response = f"Here are the latest/recent items from your database related to your question:\n\n"
                response += formatted_context
                response += "\n\nThese are the most recent entries matching your query."
            
            else:
                # Generic response with context
                response = f"Based on your question: '{user_query}'\n\n"
                response += f"I found {len(context)} relevant document(s) in your database:\n\n"
                response += formatted_context
                response += "\n\nThis information should help answer your question."
            
            # Add note about OpenAI (optional)
            response += "\n\n💡 Note: For more sophisticated AI-powered responses, you can add credits to your OpenAI account."
            
        else:
            # No context found
            response = f"I searched your MongoDB database for information related to: '{user_query}'\n\n"
            response += "However, I couldn't find specific matching documents in your database.\n\n"
            response += "This could mean:\n"
            response += "• The data might be stored with different keywords\n"
            response += "• Try rephrasing your question\n"
            response += "• The information might not be in the database yet\n\n"
            response += "Would you like me to search for something else, or would you like to add this information to the database?"
        
        return response
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get full conversation history for a session"""
        return self._get_conversation_history(session_id, limit=100)


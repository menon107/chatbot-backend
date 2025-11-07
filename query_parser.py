"""
Query parser for natural language understanding
Handles synonyms, variations, and different ways to ask about health metrics
"""
import re
import logging

logger = logging.getLogger(__name__)

# Health metric synonyms and variations
HEALTH_METRIC_MAP = {
    # Blood Pressure - Systolic
    'systolic': 'systolic',
    'systolic blood pressure': 'systolic',
    'systolic bp': 'systolic',
    'upper blood pressure': 'systolic',
    'upper bp': 'systolic',
    'top number': 'systolic',
    'first number': 'systolic',
    'high number': 'systolic',
    
    # Blood Pressure - Diastolic
    'diastolic': 'diastolic',
    'diastolic blood pressure': 'diastolic',
    'diastolic bp': 'diastolic',
    'lower blood pressure': 'diastolic',
    'lower bp': 'diastolic',
    'bottom number': 'diastolic',
    'second number': 'diastolic',
    'low number': 'diastolic',
    
    # Pulse/Heart Rate
    'pulse': 'pulse',
    'heart rate': 'pulse',
    'pulse rate': 'pulse',
    'heartbeat': 'pulse',
    'hr': 'pulse',
    'bpm': 'pulse',
    'heart beat': 'pulse',
    
    # BMI
    'bmi': 'bmi',
    'body mass index': 'bmi',
    'body mass': 'bmi',
    
    # Blood Sugar
    'blood sugar': 'fastingBloodSugar',
    'bloodsugar': 'fastingBloodSugar',
    'glucose': 'fastingBloodSugar',
    'blood glucose': 'fastingBloodSugar',
    'fasting blood sugar': 'fastingBloodSugar',
    'fasting glucose': 'fastingBloodSugar',
    'sugar level': 'fastingBloodSugar',
    'glucose level': 'fastingBloodSugar',
    
    # Cholesterol
    'cholesterol': 'totalCholesterol',
    'total cholesterol': 'totalCholesterol',
    'cholesterol level': 'totalCholesterol',
    'total chol': 'totalCholesterol',
    
    # Waist
    'waist': 'waistCircumference',
    'waist circumference': 'waistCircumference',
    'waist size': 'waistCircumference',
    'waist measurement': 'waistCircumference',
    
    # Sleep
    'sleep': 'sleepQuality',
    'sleep quality': 'sleepQuality',
    'sleep score': 'sleepQuality',
    'how i slept': 'sleepQuality',
    'sleep rating': 'sleepQuality',
    
    # Stress
    'stress': 'stressLevel',
    'stress level': 'stressLevel',
    'stress score': 'stressLevel',
    'how stressed': 'stressLevel',
    'anxiety level': 'stressLevel',
}

# Profile field synonyms
PROFILE_FIELD_MAP = {
    'height': 'height',
    'how tall': 'height',
    'tall': 'height',
    'height in cm': 'height',
    'height in centimeters': 'height',
    
    'weight': 'weight',
    'how much do i weigh': 'weight',
    'body weight': 'weight',
    'weight in kg': 'weight',
    'weight in kilograms': 'weight',
    'my weight': 'weight',
    
    'age': 'age',
    'how old': 'age',
    'my age': 'age',
    'years old': 'age',
    
    'gender': 'gender',
    'sex': 'gender',
    'male or female': 'gender',
    
    'smoker': 'smoker',
    'smoking': 'smoker',
    'do i smoke': 'smoker',
    'smoking status': 'smoker',
    
    'diabetes': 'diabetes',
    'have diabetes': 'diabetes',
    'diabetic': 'diabetes',
    'diabetes status': 'diabetes',
    
    'hypertension': 'hypertension_treated',
    'high blood pressure treated': 'hypertension_treated',
    'hypertension treated': 'hypertension_treated',
    'bp treated': 'hypertension_treated',
    'blood pressure treated': 'hypertension_treated',
    
    'family history': 'family_history_of_cardiovascular_disease',
    'family history of heart disease': 'family_history_of_cardiovascular_disease',
    'family heart disease': 'family_history_of_cardiovascular_disease',
    'cardiovascular family history': 'family_history_of_cardiovascular_disease',
    
    'atrial fibrillation': 'atrial_fibrillation',
    'afib': 'atrial_fibrillation',
    'a-fib': 'atrial_fibrillation',
    'atrial fib': 'atrial_fibrillation',
    
    'kidney disease': 'chronic_kidney_disease',
    'chronic kidney disease': 'chronic_kidney_disease',
    'ckd': 'chronic_kidney_disease',
    'kidney problems': 'chronic_kidney_disease',
    
    'arthritis': 'rheumatoid_arthritis',
    'rheumatoid arthritis': 'rheumatoid_arthritis',
    'ra': 'rheumatoid_arthritis',
    'joint problems': 'rheumatoid_arthritis',
    
    'copd': 'chronic_obstructive_pulmonary_disorder',
    'chronic obstructive pulmonary disorder': 'chronic_obstructive_pulmonary_disorder',
    'lung disease': 'chronic_obstructive_pulmonary_disorder',
    'breathing problems': 'chronic_obstructive_pulmonary_disorder',
}

# Time format synonyms
TIME_VARIATIONS = {
    'morning': ['06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00'],
    'afternoon': ['12:00', '13:00', '14:00', '15:00', '16:00', '17:00'],
    'evening': ['18:00', '19:00', '20:00', '21:00'],
    'night': ['22:00', '23:00', '00:00', '01:00', '02:00', '03:00', '04:00', '05:00'],
    'noon': ['12:00'],
    'midday': ['12:00'],
    'midnight': ['00:00'],
}

def extract_health_metric(query: str) -> str:
    """
    Extract health metric from query using synonyms
    
    Args:
        query: User query string
        
    Returns:
        Metric field name or None
    """
    query_lower = query.lower()
    
    # Check for exact matches first
    for keyword, field in HEALTH_METRIC_MAP.items():
        if keyword in query_lower:
            logger.info(f"Matched health metric: '{keyword}' -> '{field}'")
            return field
    
    return None

def extract_profile_field(query: str) -> str:
    """
    Extract profile field from query using synonyms
    
    Args:
        query: User query string
        
    Returns:
        Profile field name or None
    """
    query_lower = query.lower()
    
    # Check for exact matches first
    for keyword, field in PROFILE_FIELD_MAP.items():
        if keyword in query_lower:
            logger.info(f"Matched profile field: '{keyword}' -> '{field}'")
            return field
    
    return None

def extract_time_context(query: str) -> str:
    """
    Extract time context from query (morning, afternoon, evening, etc.)
    
    Args:
        query: User query string
        
    Returns:
        Time string or None
    """
    query_lower = query.lower()
    
    # Check for time variations
    for time_keyword, time_values in TIME_VARIATIONS.items():
        if time_keyword in query_lower:
            # Return the first time value as default
            return time_values[0]
    
    # Try to extract specific time (HH:MM format)
    time_pattern = r'\b(\d{1,2}):(\d{2})\b'
    match = re.search(time_pattern, query)
    if match:
        hour = int(match.group(1))
        minute = match.group(2)
        # Format as HH:MM
        return f"{hour:02d}:{minute}"
    
    # Try 12-hour format with AM/PM
    time_pattern_12h = r'\b(\d{1,2})\s*(am|pm|AM|PM)\b'
    match = re.search(time_pattern_12h, query_lower)
    if match:
        hour = int(match.group(1))
        period = match.group(2).lower()
        if period == 'pm' and hour != 12:
            hour += 12
        elif period == 'am' and hour == 12:
            hour = 0
        return f"{hour:02d}:00"
    
    return None

def parse_query(query: str) -> dict:
    """
    Parse user query to extract all relevant information
    
    Returns:
        Dictionary with:
        - metric: Health metric field name
        - profile_field: Profile field name
        - date: Date in YYYY-MM-DD format
        - time: Time in HH:MM format
        - is_profile_query: Boolean
        - is_date_query: Boolean
    """
    result = {
        'metric': None,
        'profile_field': None,
        'date': None,
        'time': None,
        'is_profile_query': False,
        'is_date_query': False,
    }
    
    # Extract health metric
    metric = extract_health_metric(query)
    if metric:
        result['metric'] = metric
        result['is_date_query'] = True  # Health metrics are usually date-based
    
    # Extract profile field
    profile_field = extract_profile_field(query)
    if profile_field:
        result['profile_field'] = profile_field
        result['is_profile_query'] = True
    
    # Extract time context
    time = extract_time_context(query)
    if time:
        result['time'] = time
    
    # Date extraction is handled separately in date_parser.py
    
    return result


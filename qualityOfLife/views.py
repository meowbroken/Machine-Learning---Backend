from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.conf import settings
import joblib
import os
import pandas as pd
import numpy as np
from .models import StudentInput
from .serializers import StudentInputSerializer

# Load model and feature selector
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'qol_regression_model.pkl')
SELECTOR_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'qol_feature_selector.pkl')

MODEL = joblib.load(MODEL_PATH)
SELECTOR = joblib.load(SELECTOR_PATH)

# All 85 original column names
COLUMNS = ['age_12', 'age_13', 'age_14', 'age_15', 'age_16', 'age_17', 'age_18',
           'age_19', 'age_20', 'age_21', 'age_22', 'age_23', 'age_24', 'age_25',
           'age_26', 'age_27', 'age_28', 'age_29', 'age_30', 'age_31', 'age_32',
           'age_33', 'age_34', 'age_35', 'gender_Female', 'gender_Male',
           'gender_Prefer not to say', 'grade_level_Grade 7',
           'grade_level_Grade 8', 'grade_level_Grade 9', 'grade_level_Grade 10',
           'grade_level_Grade 11', 'grade_level_Grade 12', 'grade_level_1st Year',
           'grade_level_2nd Year', 'grade_level_3rd Year', 'grade_level_4th Year',
           'grade_level_5th Year', 'strand_track_JHS', 'strand_track_STEM',
           'strand_track_ICT', 'strand_track_HUMSS', 'strand_track_ABM',
           'strand_track_BMMA', 'strand_track_BSA', 'strand_track_BSA-ANSCI',
           'strand_track_BSAMT', 'strand_track_BSARC', 'strand_track_BSBIO',
           'strand_track_BSBIO-EBIO', 'strand_track_BSCE', 'strand_track_BSCJ',
           'strand_track_BSCPE', 'strand_track_BSCS', 'strand_track_BSED',
           'strand_track_BSESS', 'strand_track_BSIT', 'strand_track_BSMLS',
           'strand_track_BSMARE', 'strand_track_BSN', 'strand_track_BSPSYCH',
           'strand_track_BSPT', 'strand_track_BSTM',
           'internet_access_Home Wi-Fi (Personal/Family Router)',
           'internet_access_Shared Wi-Fi (Dormitory, Boarding House, Hotspot)',
           'internet_access_Mobile Data',
           'internet_access_Wired/Broadband (LAN/Ethernet)',
           'internet_access_No stable internet access', 'device_type_Smartphone',
           'device_type_Tablet', 'device_type_Laptop', 'device_type_Desktop PC',
           'daily_screen_time', 'pre_sleep_use', 'usage_purpose',
           'device_check_freq', 'overuse_awareness', 'control_level',
           'usage_interrupts_tasks', 'wakeup_use', 'num_socials',
           'attempted_detox', 'aware_of_hours', 'time_of_day_use',
           'sleep_disrupted']


# === Encoders ===

def one_hot_encode(value, prefix, options):
    return {f"{prefix}_{opt}": int(str(opt) == str(value)) for opt in options}

def multi_hot_encode(values, prefix, options):
    return {f"{prefix}_{opt}": int(opt in values) for opt in options}

def score_behavioral_inputs(raw):
    screen_time_map = {
        "Less than 1 hour": 1, "1–3 hours": 2, "3–5 hours": 3,
        "5–7 hours": 4, "More than 7 hours": 5
    }
    pre_sleep_map = {
        "I don't use a device before sleeping": 1,
        "Less than 15 minutes": 2, "15–30 minutes": 3,
        "30–60 minutes": 4, "More than 1 hour": 5
    }
    time_of_day_map = {
        "Morning (6 AM – 12 PM)": 1, "Afternoon (12 PM – 6 PM)": 2,
        "Evening (6 PM – 10 PM)": 3, "Late Night (10 PM – 3 AM)": 4,
        "Equally throughout the day": 3
    }
    usage_purpose_score = {
        "Social Media": 2, "Streaming (YouTube, etc.)": 2, "Gaming": 2,
        "Online Shopping": 1, "School Work/ Studying": 0,
        "Reading / Research": 0, "Work-related tasks": 0
    }
    purpose_total = sum(usage_purpose_score.get(p, 0) for p in raw.get("usage_purpose", []))

    return {
        "daily_screen_time": screen_time_map.get(raw.get("screen_time"), 0) / 5,
        "pre_sleep_use": pre_sleep_map.get(raw.get("pre_sleep_use"), 0) / 5,
        "usage_purpose": purpose_total / 7,
        "device_check_freq": raw.get("device_check_freq", 1) / 5,
        "overuse_awareness": raw.get("usage_interrupts_tasks", 1) / 5,
        "control_level": (11 - raw.get("control_level", 1)) / 10,
        "usage_interrupts_tasks": raw.get("study_distraction", 1) / 5,
        "wakeup_use": int(raw.get("wakeup_use", False)),
        "num_socials": raw.get("num_socials", 0) / 10,
        "attempted_detox": int(raw.get("attempted_detox", False)),
        "aware_of_hours": int(not raw.get("aware_of_hours", True)),
        "time_of_day_use": time_of_day_map.get(raw.get("time_of_day_use"), 1) / 4,
        "sleep_disrupted": int(raw.get("sleep_disrupted", False))
    }

def prepare_features(data):
    features = {}

    features.update(one_hot_encode(data.get("age"), "age", [str(i) for i in range(12, 36)]))
    features.update(one_hot_encode(data.get("gender"), "gender", ["Female", "Male", "Prefer not to say"]))
    features.update(one_hot_encode(data.get("grade_level"), "grade_level", [
        "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12",
        "1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year"
    ]))
    features.update(one_hot_encode(data.get("strand"), "strand_track", [
        "JHS", "STEM", "ICT", "HUMSS", "ABM", "BMMA", "BSA", "BSA-ANSCI", "BSAMT", "BSARC", "BSBIO", "BSBIO-EBIO",
        "BSCE", "BSCJ", "BSCPE", "BSCS", "BSED", "BSESS", "BSIT", "BSMLS", "BSMARE", "BSN", "BSPSYCH", "BSPT", "BSTM"
    ]))
    features.update(multi_hot_encode(data.get("internet_access", []), "internet_access", [
        "Home Wi-Fi (Personal/Family Router)", "Mobile Data",
        "Wired/Broadband (LAN/Ethernet)", "Shared Wi-Fi (Dormitory, Boarding House, Hotspot)",
        "No stable internet access"
    ]))
    features.update(multi_hot_encode(data.get("devices", []), "device_type", [
        "Smartphone", "Tablet", "Laptop", "Desktop PC"
    ]))

    features.update(score_behavioral_inputs(data))

    # Convert to DataFrame with expected feature names
    df = pd.DataFrame([features], columns=COLUMNS)

    # Transform using feature selector
    transformed = SELECTOR.transform(df)

    return np.array(transformed)

@api_view(['POST'])
def predict_quality_of_life(request):
    serializer = StudentInputSerializer(data=request.data)
    if serializer.is_valid():
        try:
            features = prepare_features(serializer.validated_data)
            prediction = MODEL.predict(features)[0]  # normalized 0–1
            return Response({
                'predicted_qol': round(prediction * 100, 2),
                'success': True
            }, status=status.HTTP_200_OK)

        except FileNotFoundError:
            return Response({
                'error': 'Model file not found.',
                'success': False,
                'received_data': request.data
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            return Response({
                'error': f'Prediction error: {str(e)}',
                'success': False,
                'received_data': request.data
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({
        'success': False,
        'errors': serializer.errors,
        'received_data': request.data
    }, status=status.HTTP_400_BAD_REQUEST)


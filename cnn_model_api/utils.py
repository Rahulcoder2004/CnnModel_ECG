from django.db import models

# Create your models here.

# cnn_model_api/utils.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define image dimensions
img_height, img_width = 64, 64

# Class names (update if different)
class_names = ['s', 'm', 'q', 'n', 'f', 'v']

# Class descriptions
class_descriptions = {
    'f': 'Fusion of ventricular and normal beat', # Example: You might want to rename this to "Fusion Beat" in React
    'n': 'Normal', # Match this with a "Normal" entry in your heartConditions or handle it separately
    'm': 'Morphological variations',
    'q': 'Unclassifiable',
    's': 'Supraventricular premature beat', # Example: This should match the key in heartConditions
    'v': 'Premature ventricular contraction' # Example: This should match the key in heartConditions
}

# Load the model once when the Django app starts
_model = None

def load_ecg_model():
    global _model
    if _model is None:
        # Construct the absolute path to the model file
        # Assuming your model is in cnn_model_api/models/ecg_model.h5
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'ecg_model.h5')
        try:
            _model = load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            _model = None # Ensure it's None if loading fails
    return _model


def predict_ecg_image(image_file, confidence_threshold=0.7):
    model = load_ecg_model()
    if model is None:
        return "Error: Model not loaded", 0.0, None # Added None for condition

    try:
        # Read image from InMemoryUploadedFile
        img_np = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return "Error: Invalid image file", 0.0, None

        # Resize and preprocess
        img = cv2.resize(img, (img_height, img_width))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        predictions = model.predict(img, verbose=0)[0]
        max_confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)

        # Check confidence threshold
        if max_confidence < confidence_threshold:
            return "uncertain", max_confidence, "Unclassified" # Returning "uncertain" status and condition "Unclassified"

        # Return prediction
        predicted_class_key = class_names[predicted_class_index]
        predicted_condition_name = class_descriptions.get(predicted_class_key, 'Unknown Condition')
        if predicted_class_key == 'n':
            status_result = "normal"
        elif predicted_class_key == 'q':
            status_result = "uncertain"
        else:
            status_result = "abnormal"

        return status_result, max_confidence, predicted_condition_name

    except Exception as e:
        return f"Error: {str(e)}", 0.0, None


# cnn_model_api/utils.py
import os
import smtplib
from email.message import EmailMessage

# utils.py

import requests
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

def send_email_report(to_email: str, result: dict) -> bool:
    """
    Send ECG report email using Resend API.
    """

    if not settings.EMAIL_ENABLED:
        logger.info("Email sending disabled via config")
        return False

    # Prepare data for email
    status_text = result.get("status", "N/A")
    confidence = float(result.get("confidence", 0)) * 100
    condition = result.get("condition", "Not specified")
    description = result.get("description", "")

    try:
        api_key = settings.RESEND_API_KEY

        payload = {
            "from": settings.DEFAULT_FROM_EMAIL,
            "to": [to_email],
            "subject": f"ECG Report: {condition}",
            "html": f"""
            <html>
            <body>
                <h2>ECG Analysis Report</h2>
                <p><strong>Status:</strong> {status_text}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                <p><strong>Condition:</strong> {condition}</p>
                <p>{description}</p>
            </body>
            </html>
            """
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://api.resend.com/emails", json=payload, headers=headers)

        if response.status_code == 200 or response.status_code == 202:
            logger.info(f"Resend email sent to {to_email}")
            return True
        else:
            logger.error(f"Resend email failed: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Exception in send_email_report: {e}", exc_info=True)
        return False


# from django.core.mail import send_mail
# from django.conf import settings
# import logging
# from datetime import datetime

# logger = logging.getLogger(__name__)

# def send_email_report(to_email: str, result: dict) -> bool:
#     """
#     Send ECG report email (SYNC & SAFE for Railway)
#     """

#     if not settings.CUSTOM_EMAIL_ENABLED:
#         logger.warning("Email disabled by settings")
#         return False

#     try:
#         status_text = result.get("status", "N/A")
#         confidence = float(result.get("confidence", 0)) * 100
#         condition = result.get("condition", "Unknown")
#         description = result.get("description", "")

#         subject = f"ECG Report: {condition}"

#         text_message = f"""
# ECG Analysis Report

# Status: {status_text}
# Confidence: {confidence:.2f}%
# Condition: {condition}

# {description}

# This is an AI-generated report. Consult a doctor.
# """

#         html_message = f"""
#         <html>
#         <body style="font-family: Arial;">
#             <h2>ECG Analysis Report</h2>
#             <p><strong>Status:</strong> {status_text}</p>
#             <p><strong>Confidence:</strong> {confidence:.2f}%</p>
#             <p><strong>Condition:</strong> {condition}</p>
#             <p>{description}</p>
#             <hr>
#             <small>This is an automated AI report.</small>
#         </body>
#         </html>
#         """

#         send_mail(
#             subject=subject,
#             message=text_message,
#             from_email=settings.DEFAULT_FROM_EMAIL,
#             recipient_list=[to_email],
#             html_message=html_message,
#             fail_silently=False,
#         )

#         logger.info(f"Email sent to {to_email}")
#         return True

#     except Exception as e:
#         logger.error(f"Email failed: {e}", exc_info=True)
#         return False


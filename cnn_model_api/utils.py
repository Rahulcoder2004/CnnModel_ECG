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
import logging
from django.conf import settings # Import Django settings
from django.core.mail import send_mail # Django's built-in email sender
from datetime import datetime # Make sure datetime is imported at the top

logger = logging.getLogger(__name__)

# --- Start of your existing model/prediction related code ---
# Example (make sure these are defined here):
# model = load_model(...)
# class_names = [...]
# class_descriptions = {
#     'f': 'Fusion of ventricular and normal beat',
#     'n': 'Normal beat',
#     'm': 'Morphological variations',
#     'q': 'Unclassifiable beat',
#     's': 'Supraventricular premature beat',
#     'v': 'Premature ventricular contraction'
# }

# def predict_ecg_image(image_file, confidence_threshold=0.7):
#     model = load_ecg_model()
#     if model is None:
#         return "Error: Model not loaded", 0.0, None # Added None for condition

#     try:
#         # Read image from InMemoryUploadedFile
#         img_np = np.frombuffer(image_file.read(), np.uint8)
#         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

#         if img is None:
#             return "Error: Invalid image file", 0.0, None

#         # Resize and preprocess
#         img = cv2.resize(img, (img_height, img_width))
#         img = img / 255.0
#         img = np.expand_dims(img, axis=0)

#         # Make prediction
#         predictions = model.predict(img, verbose=0)[0]
#         max_confidence = np.max(predictions)
#         predicted_class_index = np.argmax(predictions)

#         # Check confidence threshold
#         if max_confidence < confidence_threshold:
#             return "uncertain", max_confidence, "Unclassified" # Returning "uncertain" status and condition "Unclassified"

#         # Return prediction
#         predicted_class_key = class_names[predicted_class_index]
#         predicted_condition_name = class_descriptions.get(predicted_class_key, 'Unknown Condition')
#         if predicted_class_key == 'n':
#             status_result = "normal"
#         elif predicted_class_key == 'q':
#             status_result = "uncertain"
#         else:
#             status_result = "abnormal"

#         return status_result, max_confidence, predicted_condition_name

#     except Exception as e:
#         return f"Error: {str(e)}", 0.0, None

# --- End of your existing model/prediction related code ---


# --- Start of send_email_report function ---
async def send_email_report(to_email: str, result: dict) -> bool:
    """Send ECG report email with error handling using Django's email backend."""
    if not settings.CUSTOM_EMAIL_ENABLED:
        logger.warning("Email sending is disabled via config (settings.CUSTOM_EMAIL_ENABLED=False)")
        return False

    try:
        status_text = result.get('status', 'N/A')
        # Ensure confidence is formatted correctly for email
        confidence_value = result.get('confidence', 0)
        confidence_text = f"{float(confidence_value)*100:.2f}%" # Convert to float before formatting

        condition_text = result.get('condition', 'Not specified')
        description_text = result.get('description', '')

        text_content = f"""
ECG Analysis Report:

Status: {status_text}
Confidence: {confidence_text}
Condition: {condition_text}
Description: {description_text}

---
Important: This is an automated report based on AI analysis of your ECG image. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper interpretation of ECG results and medical guidance.
"""
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 20px auto; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
            <div style="background-color: #2563eb; color: white; padding: 20px; text-align: center;">
                <h2 style="margin: 0; font-size: 24px;">ECG Analysis Report</h2>
            </div>
            <div style="padding: 20px;">
                <p style="font-size: 18px; margin-bottom: 15px;">Your ECG image has been analyzed:</p>
                <div style="background: #f3f4f6; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #2563eb;">
                    <p style="margin-bottom: 8px;"><strong>Status:</strong> <span style="color: { 'green' if status_text == 'normal' else 'red' if status_text == 'abnormal' else 'orange' }; font-weight: bold;">{status_text.capitalize()}</span></p>
                    <p style="margin-bottom: 8px;"><strong>Confidence:</strong> {confidence_text}</p>
                    <p style="margin-bottom: 8px;"><strong>Predicted Condition:</strong> {condition_text}</p>
                    <p style="margin-bottom: 0;"><strong>Description:</strong> {description_text}</p>
                </div>
                <p style="font-size: 14px; color: #6b7280;">
                    <em>Important: This is an automated report based on AI analysis of your ECG image. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper interpretation of ECG results and medical guidance.</em>
                </p>
            </div>
            <div style="background-color: #f0f0f0; padding: 15px; text-align: center; font-size: 12px; color: #777;">
                &copy; {datetime.now().year} Your ECG Project. All rights reserved.
            </div>
        </div>
        </body>
        </html>
        """

        send_mail(
            subject=f"ECG Report: {condition_text}",
            message=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[to_email],
            html_message=html_content,
            fail_silently=False,
        )
        logger.info(f"Email report successfully queued for {to_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email report to {to_email}: {e}", exc_info=True)
        return False
# --- End of send_email_report function ---
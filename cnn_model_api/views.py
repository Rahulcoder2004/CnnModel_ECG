# from django.shortcuts import render

# # Create your views here.
# # cnn_model_api/views.py
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.decorators import api_view, parser_classes
# from .utils import predict_ecg_image, class_descriptions, class_names # Import necessary components
# from rest_framework.response import Response
# from rest_framework import status

# # Ensure you have Django REST Framework installed: pip install djangorestframework

# @csrf_exempt # Use this for simplicity during development, but consider proper CSRF handling in production
# @api_view(['POST'])
# @parser_classes([MultiPartParser, FormParser])
# def predict_image(request):
#     if 'file' not in request.FILES:
#         return Response({'detail': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

#     image_file = request.FILES['file']
#     email = request.POST.get('email', 'N/A') # Get email if sent, otherwise N/A

#     prediction_status, confidence, predicted_condition_name = predict_ecg_image(image_file)

#     if "Error" in prediction_status:
#         return Response({'detail': prediction_status}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#     response_data = {
#         'status': prediction_status, # 'normal', 'abnormal', 'uncertain'
#         'confidence': confidence,
#         'condition': predicted_condition_name, # e.g., "Atrial Fibrillation"
#         'description': class_descriptions.get(next((k for k, v in class_descriptions.items() if v == predicted_condition_name), 'unknown'), 'No specific description available.')
#         # You might want to refine how description is mapped from class_descriptions here
#     }

#     return Response(response_data, status=status.HTTP_200_OK)


# cnn_model_api/views.py
# cnn_model_api/views.py
# cnn_model_api/views.py
# cnn_model_api/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status

import os
import shutil
import logging
import threading
import asyncio

# Import necessary components from your utils.py
# Make sure send_email_report, predict_ecg_image, and class_descriptions are correctly defined in utils.py
from .utils import predict_ecg_image, class_descriptions, send_email_report 

logger = logging.getLogger(__name__)

@csrf_exempt # Use this for simplicity during development, but consider proper CSRF handling in production
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def predict_image(request):
    # Check if a file was provided in the request
    if 'file' not in request.FILES:
        logger.error("No image file provided in request.")
        return Response({'detail': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    image_file = request.FILES['file']
    email_address = request.POST.get('email', None) # Get email, default to None if not provided

    # temp_file_path is not strictly necessary here if predict_ecg_image can handle
    # InMemoryUploadedFile directly. If your predict_ecg_image requires a file path,
    # you'd uncomment and implement saving the file to a temporary location first.
    # For now, assuming predict_ecg_image takes the in-memory file.
    temp_file_path = None 

    try:
        # Perform the ECG prediction using your model in utils.py
        # These variables will hold the ACTUAL prediction results from your model
        prediction_status, confidence, predicted_condition_name = predict_ecg_image(image_file)

        # Handle prediction errors (if 'Error' string is returned from predict_ecg_image)
        if "Error" in prediction_status:
            logger.error(f"ECG prediction failed: {prediction_status}")
            return Response({'detail': prediction_status}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Get the detailed description for the predicted condition
        # Ensure class_descriptions is imported from utils.py and properly defined there
        predicted_description = class_descriptions.get(
            next((k for k, v in class_descriptions.items() if v == predicted_condition_name), None),
            'No specific description available.'
        )

        # Prepare the response data that will be sent back to the React frontend
        response_data = {
            'status': prediction_status,          # e.g., 'normal', 'abnormal', 'uncertain'
            'confidence': confidence,             # Confidence should be 0-100 here as per previous discussion
            'condition': predicted_condition_name, # e.g., "Atrial Fibrillation"
            'description': predicted_description, # Detailed description for the frontend
        }

        # If an email address is provided, attempt to send the report
        if email_address:
            # Prepare the result dictionary to be sent in the email.
            # This ensures the email contains the ACTUAL prediction results.
            email_result = {
                'status': prediction_status,
                'confidence': confidence,
                'condition': predicted_condition_name,
                'description': predicted_description,
            }

            # Define a wrapper function to run the async send_email_report in a new thread
            # This prevents blocking the main Django request thread.
            def run_send_email_in_thread():
                try:
                    # Create a new event loop for this thread to run the async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    # Run the async email sending function
                    loop.run_until_complete(send_email_report(email_address, email_result))
                    loop.close() # Close the event loop when done
                except Exception as e:
                    logger.error(f"Error in email sending background thread: {e}", exc_info=True)

            # Start the email sending task in a separate thread
            email_thread = threading.Thread(target=run_send_email_in_thread)
            email_thread.start()
            logger.info("Email sending task started in a new thread.")

        # Return the prediction response to the frontend
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        # Catch any unexpected errors during the process and log them
        logger.error(f"Error processing ECG request: {e}", exc_info=True)
        return Response({'detail': f'Error processing ECG: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # If you were saving a temporary file, clean it up here
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

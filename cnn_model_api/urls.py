# cnn_model_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_image, name='predict_image'),
]
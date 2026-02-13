#!/bin/bash
python manage.py migrate
gunicorn Django_CnnModel.wsgi:application

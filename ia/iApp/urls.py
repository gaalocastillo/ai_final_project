#!iApp//urls.py
"""Ruteador de url con view"""
from django.urls import path
from iApp.views import *

app_name = "ControlEPP"

urlpatterns = [
    path(r'', clasificador),
]

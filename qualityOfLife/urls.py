from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_quality_of_life, name='predict-qol'),
]
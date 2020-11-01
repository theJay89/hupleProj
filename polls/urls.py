# 작성
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_upload, name = 'home_upload'),
    path('index', views.index, name='index'),
    
]
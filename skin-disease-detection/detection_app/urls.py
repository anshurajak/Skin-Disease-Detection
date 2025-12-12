from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('feedback/', views.feedback, name='feedback'),
    path('', views.home, name='home'),
    path('predict/', views.upload_image, name='predict_skin_disease'),
    # path('predict/', views.predict_disease, name='predict_disease'),
]

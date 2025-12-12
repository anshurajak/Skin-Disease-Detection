from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Doctor(models.Model):
    name = models.CharField(max_length=100)
    specialization = models.CharField(max_length=100)
    contact_info = models.TextField()

    def __str__(self):
        return self.name

class Feedback(models.Model):
    name = models.CharField(max_length=100,null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    message = models.TextField()

    def __str__(self):
        return self.name

class SkinDisease(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    image = models.ImageField(upload_to='disease_images/')

    def __str__(self):
        return self.name

class SkinDiseasePrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    disease_name = models.CharField(max_length=100, blank=True, null=True)
    causes = models.TextField(blank=True, null=True)
    grad_cam_path = models.ImageField(upload_to='grad_cam/', blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username} - {self.disease_name}"
    
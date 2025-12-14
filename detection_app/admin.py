from django.contrib import admin
from .models import SkinDisease, SkinDiseasePrediction ,Feedback

admin.site.register(SkinDisease)
admin.site.register(SkinDiseasePrediction)
admin.site.register(Feedback)
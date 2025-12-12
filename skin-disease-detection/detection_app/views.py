from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm
from .forms import ImageUploadForm, FeedbackForm, UserLoginForm
from .models import Feedback, SkinDiseasePrediction
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import logging
from django.conf import settings  # Import settings for media path
import os

# Import functions from grad_cam.py

from .model_utils import predict_skin_disease, load_trained_model

# Configure logging
logging.basicConfig(level=logging.INFO)

def home(request):
    return render(request, 'detection_app/index.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'detection_app/register.html', {'form': form})

def user_login(request):
    if request.method == "POST":
        form = UserLoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                return render(request, 'detection_app/login.html', {'form': form, 'error': 'Invalid credentials'})
    else:
        form = UserLoginForm()
    return render(request, 'detection_app/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('home')

@csrf_exempt
@login_required
def upload_image(request):
    context = {}

    if request.method == 'POST':
        logging.info(f"📤 POST data: {request.POST}")
        logging.info(f"📁 FILES data: {request.FILES}")

        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            img = request.FILES.get('image')  # Handle single image
            logging.info(f"📥 Received image: {img.name}")

            # Check file type
            if not img.name.endswith(('.png', '.jpg', '.jpeg')):
                logging.warning("⚠️ Invalid file type.")
                context['predictions'] = [{'error': 'Invalid file type. Please upload a PNG or JPG image.'}]
                return render(request, 'detection_app/prediction_results.html', context)

            try:
                # ✅ Save uploaded image
                prediction_instance = SkinDiseasePrediction(user=request.user, image=img)
                prediction_instance.save()
                img_path = prediction_instance.image.path
                logging.info(f"✅ Image saved at: {img_path}")

                # ✅ Load model and run prediction
                model = load_trained_model()
                result = predict_skin_disease(model, img_path)

                if result:
                    prediction_instance.disease_name = result['disease_name']
                    prediction_instance.confidence = result['confidence']
                    prediction_instance.save()

                    context['predictions'] = [{
                        'image_path': prediction_instance.image.url,
                        'disease_name': result['disease_name'],
                        'confidence': result['confidence'],
                        'details': result['details'],
                    }]

                    logging.info(f"🔍 Prediction result: {context['predictions']}")
                else:
                    context['predictions'] = [{'error': 'Prediction failed'}]
                    logging.warning("⚠️ Prediction returned no result")

            except Exception as e:
                logging.error(f"❌ Error during prediction: {e}")
                context['predictions'] = [{'error': 'Error processing image'}]

            # ✅ Redirect to result page
            return render(request, 'detection_app/prediction_results.html', context)

        else:
            logging.warning("⚠️ Form is not valid.")
            context['predictions'] = [{'error': 'Invalid form submission'}]

    else:
        logging.info("📄 Rendering upload form (GET request).")
        form = ImageUploadForm()

    return render(request, 'detection_app/upload_image.html', {'form': form})


from .models import Feedback

from django.shortcuts import render
from .models import Feedback

def feedback(request):
    success = ''
    if request.method == 'POST':
        name = request.POST.get('user_name')
        email = request.POST.get('user_email')
        message = request.POST.get('feedback')

        if message:  # Message is required
            Feedback.objects.create(name=name, email=email, message=message)
            success = 'Thanks for your feedback!'

    feedback_list = Feedback.objects.all().order_by('-id')
    return render(request, 'detection_app/feedback.html', {
        'success': success,
        'feedback_list': feedback_list
    })






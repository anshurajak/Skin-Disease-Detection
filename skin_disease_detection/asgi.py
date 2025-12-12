from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_detection.settings')

application = get_asgi_application()
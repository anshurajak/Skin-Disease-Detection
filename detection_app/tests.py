from django.test import TestCase
from django.urls import reverse
from .models import User, Doctor, Feedback

class UserRegistrationTest(TestCase):
    def test_user_registration(self):
        response = self.client.post(reverse('register'), {
            'username': 'testuser',
            'password': 'testpassword',
            'email': 'testuser@example.com'
        })
        self.assertEqual(response.status_code, 302)  # Redirect after successful registration
        self.assertTrue(User.objects.filter(username='testuser').exists())

class UserLoginTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpassword')

    def test_user_login(self):
        response = self.client.post(reverse('login'), {
            'username': 'testuser',
            'password': 'testpassword'
        })
        self.assertEqual(response.status_code, 302)  # Redirect after successful login

class ImageUploadTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.client.login(username='testuser', password='testpassword')

    def test_image_upload(self):
        with open('path/to/test/image.jpg', 'rb') as img:
            response = self.client.post(reverse('upload_image'), {'image': img})
        self.assertEqual(response.status_code, 200)  # Check for successful upload

class DoctorSuggestionsTest(TestCase):
    def test_doctor_suggestions(self):
        response = self.client.get(reverse('doctor_suggestions'))
        self.assertEqual(response.status_code, 200)  # Check if suggestions page loads

class FeedbackManagementTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.client.login(username='testuser', password='testpassword')

    def test_feedback_submission(self):
        response = self.client.post(reverse('feedback'), {
            'content': 'Great service!',
            'rating': 5
        })
        self.assertEqual(response.status_code, 302)  # Redirect after successful feedback submission
        self.assertTrue(Feedback.objects.filter(content='Great service!').exists())
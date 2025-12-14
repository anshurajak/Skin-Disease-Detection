# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from detection_app.model_utils import predict_skin_disease

# # Test the prediction function
# image_path = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset\rosacea\image2.png"  # Replace with the path to your test image
# result = predict_skin_disease(image_path)
# print(f"Predicted Class: {result}")

import numpy as np
from model_utils import load_trained_model, preprocess_image

# Load model
model = load_trained_model()

# List of test images
test_images = [
    "C:/Users/Anshu Kumar Rajak/Desktop/Skin Detection/skin-disease-detection/dataset/rosacea/image2.png",
    # "C:/Users/Anshu Kumar Rajak/Desktop/Skin/skin-disease-detection/media/uploads/images_1.jpg",
    # "C:/Users/Anshu Kumar Rajak/Desktop/Skin/skin-disease-detection/media/uploads/images.jpg"
]

# Class labels
class_labels = [
    "Acne", "Eczema", "Psoriasis", "Melanoma", "Rosacea", "Vitiligo",
    "Seborrheic Dermatitis", "Fungal Infection", "Basal Cell Carcinoma", "Other"
]

for img_path in test_images:
    img_array = preprocess_image(img_path)
    
    if img_array is None:
        print(f"❌ Error loading image: {img_path}")
        continue

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    predicted_label = class_labels[predicted_class]
    print(f"✅ Image: {img_path} | Predicted Disease: {predicted_label}")

import logging
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------------ Config ------------------ #
logging.basicConfig(level=logging.INFO)
MODEL_PATH = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin Detection\skin-disease-detection\models\ham_data_skin_disease_detection.keras"

CLASS_LABELS = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis-like Lesions',
    'Cellulitis',
    'Chickenpox',
    'Dermatofibroma',
    'Eczema',
    'Impetigo',
    'Melanocytic nevus',
    'Melanoma',
    'Perleche',
    'Pitted_keratolysis',
    'Pityrosporum',
    'Ringworm',
    'Rosacea',
    'Shingles',
    'acne',
    'vasculitis']

class_info = {
    'Actinic Keratosis': {
        "Cause": "Prolonged exposure to UV radiation from the sun.",
        "Symptoms": "Rough, scaly patches on the skin, often pink or red.",
        "CommonAreas": "Face, ears, neck, scalp, hands.",
        "RiskFactors": "Fair skin, age, excessive sun exposure.",
        "Treatment": "Cryotherapy, topical medications, photodynamic therapy."
    },
    'Basal Cell Carcinoma': {
        "Cause": "Uncontrolled growth of basal cells, often due to UV exposure.",
        "Symptoms": "Pearly or waxy bump, open sore, red patch.",
        "CommonAreas": "Face, ears, neck, scalp.",
        "RiskFactors": "Excessive sun exposure, fair skin, age.",
        "Treatment": "Surgical removal, topical treatments, radiation therapy."
    },
    'Benign Keratosis-like Lesions': {
        "Cause": "Non-cancerous skin growths, often due to aging or sun",
        "Symptoms": "Waxy, raised, wart-like growths on the skin.",
        "CommonAreas": "Face, chest, back.",
        "RiskFactors": "Age, sun exposure, genetics.",
        "Treatment": "Cryotherapy, curettage, laser therapy."
    },
    'Cellulitis': {
        "Cause": "Bacterial infection (Streptococcus or Staphylococcus).",
        "Symptoms": "Red, swollen, warm skin, fever.",
        "CommonAreas": "Legs, arms, face.",
        "RiskFactors": "Skin wounds, diabetes, weakened immune system.",
        "Treatment": "Oral or IV antibiotics."
    },
    'Chickenpox': {
        "Cause": "Varicella-zoster virus.",
        "Symptoms": "Itchy red blisters, fever, fatigue.",
        "CommonAreas": "Whole body.",
        "RiskFactors": "Close contact with infected person, weakened immunity.",
        "Treatment": "Antihistamines, antiviral drugs for severe cases."
    },
    'Dermatofibroma': {
        "Cause": "Benign skin growth, possibly from minor skin injuries.",
        "Symptoms": "Firm, raised nodules, often brownish in color.",
        "CommonAreas": "Legs, arms.",
        "RiskFactors": "Skin injuries, insect bites.",
        "Treatment": "Usually none; can be removed surgically if bothersome."
    },
    'Eczema': {
        "Cause": "Overactive immune system reaction to irritants.",
        "Symptoms": "Dry, itchy, inflamed skin, blisters.",
        "CommonAreas": "Hands, feet, elbows, behind the knees.",
        "RiskFactors": "Allergies, asthma, stress, irritants.",
        "Treatment": "Moisturizers, corticosteroids, antihistamines."
    },
    'Impetigo': {
        "Cause": "Bacterial infection (Staphylococcus or Streptococcus).",
        "Symptoms": "Red sores, honey-colored crusts.",
        "CommonAreas": "Face (around nose and mouth), hands.",
        "RiskFactors": "Poor hygiene, warm climate, skin injuries.",
        "Treatment": "Antibiotics (topical or oral)."
    },
    'Melanocytic nevus': {
        "Cause": "Benign growth of melanocytes (pigment-producing cells).",
        "Symptoms": "Moles that are usually brown or black, can be flat or raised.",
        "CommonAreas": "Anywhere on the body.",
        "RiskFactors": "Genetics, sun exposure.",
        "Treatment": "Usually none; monitor for changes that may indicate malignancy."
    },
    'Melanoma': {
        "Cause": "Uncontrolled growth of melanocytes, often due to UV exposure.",
        "Symptoms": "Irregular moles with asymmetry, border changes, color variations.",
        "CommonAreas": "Sun-exposed skin (face, arms, back).",
        "RiskFactors": "Fair skin, UV exposure, family history.",
        "Treatment": "Surgery, chemotherapy, radiation, immunotherapy."
    },
    'Perleche': {
        "Cause": "Yeast or bacterial infection in mouth corners.",
        "Symptoms": "Cracks, redness, pain at mouth corners.",
        "CommonAreas": "Lips.",
        "RiskFactors": "Dry lips, dentures, licking lips, vitamin deficiencies.",
        "Treatment": "Antifungal creams, lip balm, vitamin supplementation."
    },
    'Pitted_keratolysis': {
        "Cause": "Bacterial infection from prolonged moisture.",
        "Symptoms": "Small pits in soles of feet, bad odor.",
        "CommonAreas": "Feet.",
        "RiskFactors": "Sweaty feet, tight shoes.",
        "Treatment": "Antibacterial creams, foot hygiene."
    },
    'Pityrosporum': {
        "Cause": "Yeast infection of hair follicles.",
        "Symptoms": "Itchy, acne-like bumps on chest and back.",
        "CommonAreas": "Upper back, chest, shoulders.",
        "RiskFactors": "Oily skin, heat, sweating.",
        "Treatment": "Antifungal medications, skincare adjustments."
    },
    'Ringworm': {
        "Cause": "Fungal infection.",
        "Symptoms": "Circular, red, scaly patches with clear center.",
        "CommonAreas": "Arms, legs, torso.",
        "RiskFactors": "Skin contact, shared objects, warm climate.",
        "Treatment": "Topical antifungals, oral antifungals."
    },
    'Rosacea': {
        "Cause": "Unknown; immune response, genetics, or environmental triggers.",
        "Symptoms": "Facial redness, visible blood vessels, swollen red bumps.",
        "CommonAreas": "Nose, cheeks, forehead, chin.",
        "RiskFactors": "Sun exposure, spicy food, alcohol, stress.",
        "Treatment": "Topical antibiotics, laser therapy."
    },
    'Shingles': {
        "Cause": "Reactivation of the chickenpox virus.",
        "Symptoms": "Painful rash with blisters, burning sensation.",
        "CommonAreas": "One side of the body, often torso.",
        "RiskFactors": "Age, weakened immunity, stress.",
        "Treatment": "Antiviral drugs, pain relief medications."
    },
    'acne': {
        "Cause": "Overproduction of oil, clogged pores, bacteria, inflammation.",
        "Symptoms": "Pimples, blackheads, whiteheads, nodules, cysts.",
        "CommonAreas": "Face, back, chest, shoulders.",
        "RiskFactors": "Hormonal changes, stress, diet, genetics.",
        "Treatment": "Topical creams, antibiotics, retinoids, laser therapy."
    },
    'vasculitis': {
        "Cause": "Inflammation of blood vessels.",
        "Symptoms": "Red or purple spots, ulcers, pain, swelling.",
        "CommonAreas": "Legs, arms, other body parts.",
        "RiskFactors": "Autoimmune diseases, infections, medications.",
        "Treatment": "Corticosteroids, immunosuppressive drugs."
    }
}


# ------------------ Load Keras Model ------------------ #
def load_trained_model():
    logging.info(f"📥 Loading Keras model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("✅ Keras model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading Keras model: {e}")
        return None

# ------------------ Improved Skin Detection ------------------ #
def is_skin_present(img_array, threshold=0.03):
    """
    Improved skin detection using YCrCb and HSV color space.
    Returns True if skin pixels exceed threshold ratio.
    """
    if img_array.shape[2] != 3:
        return False

    # Convert to YCrCb and HSV
    img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Skin color range for YCrCb
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    # Skin color range for HSV
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

    # Generate masks
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # Combine masks
    combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    skin_ratio = np.sum(combined_mask > 0) / (img_array.shape[0] * img_array.shape[1])

    logging.info(f"🔍 Skin pixel ratio: {skin_ratio:.4f}")
    return skin_ratio > threshold

# ------------------ Predict Function ------------------ #
def predict_skin_disease(model, image_path):
    if model is None:
        logging.error("❌ Model not available for prediction.")
        return {'error': "Model not available for prediction."}

    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(160, 160))
        img_array = img_to_array(img).astype(np.uint8)

        # --- Skin detection step ---
        if not is_skin_present(img_array):
            logging.warning("⚠️ No skin/body part detected in the image.")
            return {'error': "Please upload correct image."}

        # Preprocess input
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        predicted_label = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown"

        logging.info(f"✅ Prediction: {predicted_label} (Confidence: {confidence:.2f})")
        key = predicted_label.replace("_", " ").strip().lower()

        details = next(
            (v for k, v in class_info.items() if k.lower() == key),
            {}
        )

        return {
            'disease_name': predicted_label,
            'confidence': round(confidence * 100, 2),
            'details': details
        }


        # return {
        #     'disease_name': predicted_label,
        #     'confidence': round(confidence * 100, 2),
        #     'details': class_info.get(predicted_label.replace("_", " ").title(), {})
        # }

    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        return {'error': "Prediction failed."}

# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    image_path = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin Detection\skin-disease-detection\dataset\acne\image2.png"
    model = load_trained_model()
    result = predict_skin_disease(model, image_path)

    if 'error' in result:
        print(result['error'])
    else:
        print(f"Predicted Disease: {result['disease_name']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Details: {result['details']}")

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('weather.h5', compile=False)

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Ensure uploads directory exists
uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle the image upload and make predictions
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        if not f:
            return jsonify({'error': 'No file uploaded'}), 400

        filepath = os.path.join(uploads_dir, f.filename)
        try:
            f.save(filepath)  # Save the uploaded file
            img = Image.open(filepath)  # Open the image
            processed_img = preprocess_image(img)  # Preprocess the image
            pred = model.predict(processed_img)  # Make predictions

            # Define the mapping of class indices to weather types
            weather_dict = {'cloudy': 0, 'foggy': 1, 'rainy': 2, 'shine': 3, 'sunrise': 4}
            pred_class = np.argmax(pred, axis=1)  # Get the predicted class index

            # Get the corresponding weather type
            weather_type = [key for key, value in weather_dict.items() if value == pred_class[0]]
            weather = f"Given image comes under {weather_type[0]} weather classification"

            return jsonify({'weather': weather})
        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        finally:
            # Optionally delete the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

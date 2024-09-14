from flask import Flask, render_template, request, jsonify
import json
import os
from keras.preprocessing import image
import numpy as np
import keras

app = Flask(__name__)

# Load the saved model
loaded_model = keras.models.load_model('brain_tumor_detection_mod.h5')

# Function to load the model
def load_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = keras.models.load_model('brain_tumor_detection_mod.h5')

# Function to save message to a JSON file
def save_message(data):
    with open('messages.json', 'a') as file:
        json.dump(data, file)
        file.write('\n')

# Define a function to make predictions on a single image
def predict_single_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale to match the preprocessing used during training

    prediction = loaded_model.predict(img_array)

    if prediction[0][0] > 0.5:
        return " No Brain tumor Detected"
    else:
        return "Brain Tumor Detected"

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', current_page='home')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()  # Load the model
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # Save the file to a temporary location
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        # Make prediction on the input image
        result = predict_single_image(file_path)
        os.remove(file_path)  # Remove the temporary file

        return result, 200
    else:
        return 'Error occurred while processing the file', 500
    
@app.route('/contact', methods=['POST'])
def contact():
    if request.method == 'POST':
        data = {
            'name': request.form['name'],
            'email': request.form['email'],
            'message': request.form['message']
        }
        print("Received data:", data)  # Debugging line
        save_message(data)
        return jsonify({'message': 'Message Sent!'}), 200
    else:
        return jsonify({'error': 'Method not allowed'}), 405

@app.route('/about')
def about():
    return render_template('about.html', current_page='about')

@app.route('/learn')
def learn_more():
    return render_template('learn.html', current_page='learn')

@app.route('/contact')
def contact_form():
    return render_template('contact.html', current_page='contact')

if __name__ == '__main__':
    app.run(debug=True)

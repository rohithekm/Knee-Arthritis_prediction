from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from keras.layers import LeakyReLU
from keras.losses import SparseCategoricalCrossentropy
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024 

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Custom loss function to handle deserialization
def sparse_categorical_crossentropy_loss(from_logits=False, ignore_class=None):
    return SparseCategoricalCrossentropy(from_logits=from_logits, ignore_class=ignore_class)

# Load model and custom_objects
model = load_model('knee_model.h5', custom_objects={
    'LeakyReLU': LeakyReLU,
    'SparseCategoricalCrossentropy': sparse_categorical_crossentropy_loss
})

def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"File saved to {file_path}")
            img = prepare_image(file_path)
            prediction = model.predict(img)
            classes = ['normal', 'doubtful', 'mild', 'moderate', 'severe']
            result = classes[np.argmax(prediction)]
            return render_template('result.html', result=result)
        else:
            print("No file part")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

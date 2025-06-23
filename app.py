from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Cargar el modelo UNA SOLA VEZ
model = load_model("modelo_mildiu_mobilenet.h5")

# Clases del modelo
class_names = [
    'Hoja Sana / Healthy Leaf',
    '1 a 25% área infectada / 1 to 25% infected',
    '25 a 50% área infectada / 25 to 50% infected',
    '50 a 75% área infectada / 50 to 75% infected',
    '75 a 100% área infectada / 75 to 100% infected'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Leer imagen desde memoria sin guardarla
            img = Image.open(BytesIO(file.read())).convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predecir
            prediction_index = np.argmax(model.predict(img_array))
            prediction = class_names[prediction_index]

    return render_template("index.html", prediction=prediction)

# Ruta adicional si tienes una descripción de modelos
@app.route('/descripcion-modelos')
def descripcion_modelos():
    return render_template("descripcion_modelos.html")

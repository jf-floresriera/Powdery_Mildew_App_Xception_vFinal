from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import base64

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
    confidence = None
    image_data = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Leer imagen y procesarla
            img = Image.open(BytesIO(file.read())).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predicción
            prediction_probs = model.predict(img_array)[0]
            prediction_index = np.argmax(prediction_probs)
            prediction = class_names[prediction_index]
            confidence = round(float(prediction_probs[prediction_index]) * 100, 2)  # % con 2 decimales

            # Convertir imagen a base64 para mostrarla en HTML
            img.thumbnail((300, 300))  # Redimensionar para hacerla liviana
            buffered = BytesIO()
            img.save(buffered, format="JPEG", optimize=True, quality=70)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{img_base64}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_data=image_data
    )

@app.route('/descripcion-modelos')
def descripcion_modelos():
    return render_template("descripcion_modelos.html")

@app.route('/modelo-en-entrenamiento')
def modelo_en_entrenamiento():
    return render_template("modelo_en_entrenamiento.html")

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time

app = Flask(__name__)

# Cargar el modelo
model = load_model("modelo_mildiu_mobilenet.h5")

# Nombres de clases
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
    probs_dict = None
    top_predictions = None
    inference_time = None
    original_size = None
    entropy = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(BytesIO(file.read())).convert('RGB')
            original_size = img.size

            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Tiempo de inferencia
            start = time.time()
            prediction_probs = model.predict(img_array)[0]
            inference_time = round(time.time() - start, 4)

            # Resultados
            prediction_index = np.argmax(prediction_probs)
            prediction = class_names[prediction_index]
            confidence = round(float(prediction_probs[prediction_index]) * 100, 2)

            # Top 3
            top_indices = prediction_probs.argsort()[-3:][::-1]
            top_predictions = [(class_names[i], round(float(prediction_probs[i]) * 100, 2)) for i in top_indices]

            # Todas las probabilidades
            probs_dict = {
                class_names[i]: round(float(prediction_probs[i]) * 100, 2)
                for i in range(len(class_names))
            }

            # Entropía de la predicción
            entropy = round(-np.sum(prediction_probs * np.log(prediction_probs + 1e-10)), 4)

            # Convertir imagen cargada a base64 (reducida para ahorrar datos)
            img.thumbnail((300, 300))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", optimize=True, quality=70)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{img_base64}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_data=image_data,
        probs_dict=probs_dict,
        top_predictions=top_predictions,
        inference_time=inference_time,
        original_size=original_size,
        entropy=entropy
    )

@app.route('/descripcion-modelos')
def descripcion_modelos():
    return render_template("descripcion_modelos.html")

@app.route('/otro-modelo')
def otro_modelo():
    return render_template("otro_modelo.html")

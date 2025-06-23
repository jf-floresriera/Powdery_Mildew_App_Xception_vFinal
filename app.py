from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Cargar modelos
modelo_mobilenet = load_model("modelo_mildiu_mobilenet.h5")
modelo_xception = load_model("modelos/modelo_xception.h5")

# Clases
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
            # Leer imagen desde memoria
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction_index = np.argmax(modelo_mobilenet.predict(img_array))
            prediction = class_names[prediction_index]

    return render_template("index.html", prediction=prediction)


@app.route('/otro-modelo', methods=['GET', 'POST'])
def otro_modelo():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape((1, 224, 224, 3))

            prediction = modelo_xception.predict(img_array)
            predicted_class = np.argmax(prediction)
            result = class_names[predicted_class]

    return render_template("otro_modelo.html", result=result)


@app.route('/descripcion-modelos')
def descripcion_modelos():
    return render_template("descripcion_modelos.html")


if __name__ == '__main__':
    app.run(debug=True)




from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Solo un modelo (ligero)
model = load_model("modelo_mildiu_mobilenet.h5")

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
            # Procesar imagen directamente desde memoria
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred)
            prediction = class_names[pred_class]

    return render_template("index.html", prediction=prediction)

@app.route('/descripcion-modelos')
def descripcion_modelos():
    return render_template("descripcion_modelos.html")

if __name__ == '__main__':
    app.run(debug=True)





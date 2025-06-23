
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("modelo_mildiu_mobilenet.h5")
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
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}.jpeg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction_index = np.argmax(model.predict(img_array))
            prediction = class_names[prediction_index]
            image_url = url_for('static', filename=f"uploads/{filename}")

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)

########
#@app.route('/otro-modelo')
#def otro_modelo():
    #return render_template("otro_modelo.html")
######
@app.route('/descripcion-modelos')
def descripcion_modelos():
    return render_template("descripcion_modelos.html")


@app.route('/otro-modelo', methods=["GET", "POST"])
def otro_modelo():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join("static/uploads", file.filename)
            file.save(filepath)

            model = tf.keras.models.load_model("modelos/modelo_xception.h5")

            img = Image.open(filepath).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape((1, 224, 224, 3))

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            classes = [
                'Hoja Sana / Healthy Leaf',
                '1 a 25% área infectada / 1-25% infected',
                '25 a 50% área infectada / 25-50% infected',
                '50 a 75% área infectada / 50-75% infected',
                '75 a 100% área infectada / 75-100% infected'
            ]
            result = classes[predicted_class]

            return render_template("otro_modelo.html", result=result, filename=file.filename)
    
    return render_template("otro_modelo.html", result=None)



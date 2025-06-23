# 🌿 Powdery Mildew Detection App with Xception and MobileNet (vFinal)

> 🚀 Deep learning-based image classification tool for early detection of *powdery mildew* in crops. Deployed as a Flask web application.

## 🧪 Description | Descripción

**EN:**  
This project implements two Convolutional Neural Network (CNN) architectures, **Xception** and **MobileNet**, for detecting the presence and severity of **powdery mildew** on plant leaves. The models were trained on a custom dataset and deployed as a web application using Flask.

**ES:**  
Este proyecto implementa dos arquitecturas de redes neuronales convolucionales (CNN), **Xception** y **MobileNet**, para detectar la presencia y severidad del **mildiu polvoriento** en hojas de plantas. Los modelos fueron entrenados con un conjunto de datos personalizado y desplegados como una aplicación web usando Flask.

---

## 🧠 Models | Modelos

- `Xception`: Suitable for higher accuracy and deeper feature extraction.
- `MobileNet`: Lightweight and optimized for mobile or low-resource environments.

Both models were trained and evaluated to compare:
- Accuracy
- Inference speed
- Generalization on test images

---

## 🛠️ Technologies | Tecnologías

- Python 3.10+
- TensorFlow / Keras
- Xception & MobileNet (pretrained on ImageNet)
- Flask
- HTML/CSS
- Render (for deployment)

---

## ⚙️ Installation | Instalación

1. **Clone the repo:**
```bash
git clone https://github.com/jf-floresriera/Powdery_Mildew_App_Xception_vFinal.git
cd Powdery_Mildew_App_Xception_vFinal

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

python app.py


🌐 Deployment | Despliegue
✔️ Requirements for Render:
✅ app.py (or main.py)

✅ requirements.txt

✅ Procfile

✅ Optional: templates/ and static/ folders for UI

📤 Deploy on Render:
Go to https://render.com

Create a new Web Service

Connect your GitHub repo

Set:

Build Command: pip install -r requirements.txt

Start Command: python app.py

Environment: Python 3.10

Click Deploy

Your app will be live in a few minutes 🎉

📸 Dataset
The dataset contains labeled images of leaves with powdery mildew at 5 levels of severity.

Classes: clase0, clase1, clase2, clase3, clase4

Augmented with data augmentation techniques (rotation, zoom, brightness).

⚠️ Scientific Disclaimer | Advertencia científica
EN:
This tool is for educational and research purposes only. It is not intended to replace professional diagnosis in agricultural management.

ES:
Esta herramienta es solo para fines educativos y de investigación. No sustituye el diagnóstico profesional en el manejo agrícola.

📬 Contact
Jesús Enrique Flores Riera
Email: [jfloresr@unal.edu.co]
GitHub: jf-floresriera

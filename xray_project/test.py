from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Modeli yükle
model = load_model("xray_model.h5")

# Test görüntüsünün yolu
img_path = "datasets\\test\\PNEUMONIA\\SARS-10.1148rg.242035193-g04mr34g09a-Fig9a-day17.jpeg" # Örnek bir X-ray görüntüsü
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Tahmin yap
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Sonuç: Hasta")
else:
    print("Sonuç: Sağlıklı")
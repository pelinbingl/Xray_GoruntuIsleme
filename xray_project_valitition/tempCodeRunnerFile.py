from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Modeli yükle
model = load_model("\\xray_project\\.model.h5")

# Test görüntüsünün yolu
img_path ="\\xray_project\\datasets\\test\\NORMAL\\NORMAL2-IM-0035-0001.jpeg"  # Örnek bir X-ray görüntüsü
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Tahmin yap
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Sonuç: Hasta")
else:
    print("Sonuç: Sağlıklı")
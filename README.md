# Derin Öğrenme ile X-Ray Görüntü Sınıflandırma Projesi

## Proje Özeti
**Amaç:** X-Ray görüntülerini kullanarak pnömoni (zatürre) teşhisi için ikili sınıflandırma modeli geliştirmek.  
**Model:** Konvolüyonel Sinir Ağı (CNN)  
**Programlama Dili:** Python  
**Kütüphaneler:** TensorFlow, Keras, Matplotlib  
**Veri Seti:** Kaggle'dan alınan [Chest X-Ray Images (Pneumonia)] veri seti.

## Veri Seti Hakkında
Kullanılan veri seti, pnömoni teşhisi için etiketlenmiş 5,856 doğrulanmış göğüs röntgeni görüntüsünden oluşmaktadır. Görüntüler, bağımsız hastalardan alınmış olup eğitim ve test seti olarak ayrılmıştır.

## Veri Hazırlama
Veri seti, eğitim ve test olarak ikiye ayrılmıştır.  
Eğitim verisi, %80 eğitim ve %20 doğrulama olarak bölünmüştür.  
Görüntüler, modelin daha iyi öğrenmesi için 224x224 boyutuna getirilmiş ve piksel değerleri 1/255 ile normalize edilmiştir.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
```

## Model Mimarisi
Model, aşağıdaki katmanlardan oluşmaktadır:

1. **Konvolüyonel Katman:** 32 filtre, 3x3 çekirdek boyutu, ReLU aktivasyon fonksiyonu  
2. **Maksimum Havuzlama Katmanı:** 2x2 havuzlama boyutu  
3. **Konvolüyonel Katman:** 64 filtre, 3x3 çekirdek boyutu, ReLU aktivasyon fonksiyonu  
4. **Maksimum Havuzlama Katmanı:** 2x2 havuzlama boyutu  
5. **Düzleştirme Katmanı**  
6. **Tam Bağlantılı Katman:** 128 nöron, ReLU aktivasyon fonksiyonu  
7. **Çıkış Katmanı:** 1 nöron, sigmoid aktivasyon fonksiyonu (ikili sınıflandırma için)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Modelin Derlenmesi
Model, **Adam** optimizasyon algoritması, **binary_crossentropy** kayıp fonksiyonu ve **accuracy** metriği kullanılarak derlenmiştir.

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## Modelin Eğitimi
Model, 10 epoch boyunca eğitim verisi üzerinde eğitilmiş ve doğrulama verisi ile doğrulanmıştır.

```python
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

## Model Performansı
Eğitim süreci sonunda elde edilen doğruluk oranları:

- **Eğitim Doğruluğu:** %98.5  
- **Doğrulama Doğruluğu:** %96.43  

Test veri seti üzerinde modelin performansı değerlendirilmiştir:

- **Test Kıyıp Değeri:** 0.0073  
- **Test Doğruluk Oranı:** %100

## Sonuçların Görselleştirilmesi
Eğitim ve doğrulama süreçlerindeki doğruluk oranları aşağıdaki grafiklerde gösterilmiştir:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk Oranı')
plt.legend()
plt.title('Eğitim ve Doğrulama Doğruluk Oranları')
plt.show()
```

## Modelin Kaydedilmesi
Eğitim tamamlandıktan sonra model **.h5** formatında kaydedilmiştir.

```python
model.save("C:/Users/bingl/OneDrive/Masaüstü/xray_model.h5")
print("Model başarıyla kaydedildi!")
```


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

# Test veri seti için ayrı şekilde normalize edilmiş ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
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

## Train ve Test Datasetinden Görüntüler
Aşağıda, eğitim ve test veri setlerinden rastgele seçilen görüntüler yer almaktadır. 

### Eğitim Verisi

- **Normal:**

![Eğitim Normal Görüntüsü]![IM-0003-0001](https://github.com/user-attachments/assets/ffef49db-5702-4b40-9940-936d2bf59ca8)


- **Hasta:**

![Eğitim Hasta Görüntüsü]![1-s2 0-S1684118220300608-main pdf-002](https://github.com/user-attachments/assets/bb07d14b-3c8e-4a03-a1c3-60d7041cf2d5)


### Test Verisi

- **Normal:**

![Test Normal Görüntüsü](https://github.com/user-attachments/assets/test_normal_sample.jpg)

- **Hasta:**

![Test Hasta Görüntüsü](https://github.com/user-attachments/assets/test_pneumonia_sample.jpg)


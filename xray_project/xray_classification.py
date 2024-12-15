import tensorflow as tf
from tensorflow.keras.utils.image_dataset_from_directorye import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Dataset'in yolu
train_dir ="C:/Users/bingl/OneDrive/Masaüstü/Yeni klasör (2)/datasets/train" 
test_dir = "C:/Users/bingl/OneDrive/Masaüstü/Yeni klasör (2)/datasets/test"

# 2. Veriyi hazırlama
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

# 3. Modeli oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Modeli eğitme
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# 5. Test verisiyle doğrulama
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

results = model.evaluate(test_data)
print("Test Sonuçları:", results)

# 6. Sonuçları görselleştir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.show()
# Modeli kaydet
model.save("C:/Users/bingl/OneDrive/Masaüstü/xray_model.h5")
print("Model başarıyla kaydedildi!")
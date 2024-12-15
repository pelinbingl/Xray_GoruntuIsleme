import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 1. Dataset'in yolu (lütfen kendi dosya yolunuzu buraya ekleyin)
train_dir ="\\xray_project\\datasets\\train"
test_dir = "\\xray_project\\datasets\\test"

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

# KFold işlemi için verileri NumPy dizilerine dönüştürme
train_images, train_labels = [], []
for images, labels in train_data:
    train_images.append(images)
    train_labels.append(labels)

train_images = np.concatenate(train_images, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

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

# 4. KFold işlemi
kf = KFold(n_splits=2, shuffle=True, random_state=42)

for train_index, val_index in kf.split(train_images):
    X_train, X_val = train_images[train_index], train_images[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    # Modeli eğitme
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32
    )

    # Test verisiyle doğrulama
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    results = model.evaluate(test_data)
    print("Test Sonuçları:", results)

    # 5. Sonuçları görselleştir
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.legend()
    plt.show()

# Modeli kaydet
model.save("C:\\")  # Modeli kaydedin
print("Model başarıyla kaydedildi!")
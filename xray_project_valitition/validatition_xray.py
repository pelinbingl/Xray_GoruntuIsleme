import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 1. Dataset'in yolu (lütfen kendi dosya yolunuzu buraya ekleyin)
train_dir = "\\xray_project\\datasets\\train"
test_dir = "\\xray_project\\datasets\\test"

# 2. Veriyi hazırlama (train ve test verisi için ImageDataGenerator kullanıyoruz)
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Verileri dosyalardan alıyoruz
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

# 4. KFold işlemi (Katlama sayısını 2 olarak değiştiriyoruz)
kf = KFold(n_splits=2, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(train_data.filenames), start=1):
    print(f"Fold {fold}:")

    # KFold ile belirtilen indeksleri kullanarak train ve validation verilerini ayırma
    train_filenames = np.array(train_data.filenames)[train_index]
    val_filenames = np.array(train_data.filenames)[val_index]
    
    # ImageDataGenerator ve flow_from_directory parametrelerine yeni dosya listelerini veriyoruz
    train_data_subset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_data_subset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Eğitim işlemini başlatma
    history = model.fit(
        train_data_subset,
        validation_data=val_data_subset,
        epochs=10
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

    # 5. Sonuçları görselleştirme
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.legend()
    plt.show()

# Modeli kaydet
# Modeli kaydet
model.save("\\xray_project\\.model.h5")  # Modeli h5 formatında kaydedin
print("Model başarıyla kaydedildi!")
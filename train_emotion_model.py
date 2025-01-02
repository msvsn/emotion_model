import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import os
import matplotlib.pyplot as plt

def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # Перший згортковий блок
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Другий згортковий блок
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Третій згортковий блок
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Повнозв'язні шари
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_data_generators(train_dir, val_dir, batch_size=32):
    # Аугментація для тренувальних даних
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    # Тільки нормалізація для валідаційних даних
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Створюємо генератори
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def plot_training_history(history):
    # Графіки точності
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Точність моделі')
    plt.ylabel('Точність')
    plt.xlabel('Епоха')
    plt.legend(['Тренування', 'Валідація'], loc='lower right')
    
    # Графіки втрат
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Втрати моделі')
    plt.ylabel('Втрати')
    plt.xlabel('Епоха')
    plt.legend(['Тренування', 'Валідація'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_emotion_model(model, train_generator, validation_generator, epochs=30):
    # Компілюємо модель
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Створюємо директорію для чекпоінтів
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # Callbacks
    callbacks = [
        # Рання зупинка
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        # Зберігання найкращої моделі
        ModelCheckpoint(
            'checkpoints/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Зменшення learning rate при плато
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        # TensorBoard для візуалізації
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1,  # Зберігати гістограми ваг
            write_graph=True,  # Зберігати граф моделі
            write_images=True  # Зберігати зображення моделі
        )
    ]
    
    # Тренуємо модель
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    # Шляхи до даних
    train_dir = 'dataset/train'
    val_dir = 'dataset/test'
    
    print("Створення моделі...")
    model = create_emotion_model()
    model.summary()
    
    print("\nПідготовка генераторів даних...")
    train_generator, validation_generator = prepare_data_generators(train_dir, val_dir)
    
    print("\nПочаток тренування...")
    history = train_emotion_model(model, train_generator, validation_generator)
    
    # Зберігаємо фінальну модель
    model.save('emotion_model.keras')
    
    # Створюємо графіки
    plot_training_history(history)
    
    # Виводимо результати
    print("\nРезультати тренування:")
    print(f"Фінальна точність на тренувальному наборі: {history.history['accuracy'][-1]:.4f}")
    print(f"Фінальна точність на валідації: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main() 
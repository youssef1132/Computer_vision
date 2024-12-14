import os
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tf_keras.models import Sequential, load_model
from tf_keras.preprocessing.image import ImageDataGenerator

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 20  
train_path = 'dataset/train'
validation_path = 'dataset/validation'

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40, 
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

validation_data = validation_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

cnn_model_path = 'CNN_model.h5'
knn_model_path = 'KNN_model.pkl'

if os.path.exists(cnn_model_path):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_path)
else:
    print("Training CNN model...")
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, epochs=epochs, validation_data=validation_data)

    cnn_model.save(cnn_model_path)
    print("CNN model saved.")

def extract_features(model, data):
    features = []
    labels = []
    for _ in range(len(data)):
        imgs, lbls = next(data)
        feature_vectors = model.predict(imgs)
        features.extend(feature_vectors)
        labels.extend(lbls)
    return np.array(features), np.array(labels)

train_features, train_labels = extract_features(cnn_model, train_data)

validation_features, validation_labels = extract_features(cnn_model, validation_data)

if os.path.exists(knn_model_path):
    print("Loading pre-trained KNN model...")
    knn = joblib.load(knn_model_path)
else:
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
    knn.fit(train_features, train_labels)
    
    joblib.dump(knn, knn_model_path)
    print("KNN model saved.")

y_pred = knn.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"KNN model accuracy : {accuracy * 100:.2f}%")

random_indices = random.sample(range(len(validation_labels)), 10)

sample_images = []
real_labels = []
predicted_labels = []

validation_data.reset() 
for idx in random_indices:
    imgs, lbls = next(validation_data)
    sample_images.append(imgs[idx % batch_size])
    real_labels.append(lbls[idx % batch_size])
    predicted_labels.append(y_pred[idx])

plt.figure(figsize=(12, 8))
for i, (image, real, predicted) in enumerate(zip(sample_images, real_labels, predicted_labels)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(f"Real: {'Cat' if real == 0 else 'Dog'}\nPred: {'Cat' if predicted == 0 else 'Dog'}")
    plt.axis('off')

plt.tight_layout()
plt.show()
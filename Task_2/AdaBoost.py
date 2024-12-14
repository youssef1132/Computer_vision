import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tf_keras.models import Sequential, load_model, Model
from tf_keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

img_size = (128, 128)
batch_size = 32
epochs = 10 
train_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/train'
validation_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test'

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
    class_mode='categorical',
    shuffle=False
)

validation_data = validation_datagen.flow_from_directory(
    validation_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

cnn_model_path = 'CNN_model.h5'
ada_classifier_model_path = 'ada_classifier_model.pkl'

if os.path.exists(cnn_model_path):
    print("Loading pre-trained CNN model...")
    cnn_model = load_model(cnn_model_path)
else:
    print("Training CNN model...")
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax') 
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(train_data, epochs=epochs, validation_data=validation_data)
    
    cnn_model.save(cnn_model_path)
    print("CNN model saved.")

feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

def extract_features(model, data):
    features = []
    labels = []
    for _ in range(len(data)):
        imgs, lbls = next(data)
        feature_vectors = model.predict(imgs)
        features.extend(feature_vectors)
        labels.extend(lbls)
    return np.array(features), np.array(labels)

train_features, train_labels = extract_features(feature_extractor, train_data)

validation_features, validation_labels = extract_features(feature_extractor, validation_data)

validation_labels = np.argmax(validation_labels, axis=1)

if os.path.exists(ada_classifier_model_path):
    print("Loading pre-trained AdaBoost classifier model...")
    ada_classifier = joblib.load(ada_classifier_model_path)
else:
    print("Training AdaBoost classifier model...")
    ada_classifier = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
    ada_classifier.fit(train_features, np.argmax(train_labels, axis=1)) 
    
    joblib.dump(ada_classifier, ada_classifier_model_path)
    print("AdaBoost classifier model saved.")

y_pred = ada_classifier.predict(validation_features)
accuracy = accuracy_score(validation_labels, y_pred)
print(f"AdaBoost classifier model accuracy : {accuracy * 100:.2f}%")

def classify_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0
    features = feature_extractor.predict(img_array)
    
    prediction = ada_classifier.predict(features)
    
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    result = class_labels[prediction[0]]
    return result

img_path = 'Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/26 (66).jpg'
result = classify_image(img_path)
print(f"The image is classified as: {result}")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

print(" Loading exoplanet dataset...")
df = pd.read_csv('exoTrain.csv')

y = df['LABEL'] - 1
X = df.drop('LABEL', axis=1).values

print(" Applying Row-Wise Normalization...")
# Subtract the mean and divide by standard deviation for EACH individual star
X_scaled = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

print(" Reshaping data for 1D-CNN...")
X_reshaped = np.expand_dims(X_scaled, axis=2)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.15, random_state=42)

# Calculate class weights so the network penalizes missing a rare exoplanet
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print(" Building robust 1D-CNN architecture...")
model = Sequential([
    Conv1D(filters=16, kernel_size=7, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=4),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5), 
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print(" Training model with balanced class weights...")
model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights)

model.save('exo_model.keras')
print(" New robust model successfully saved to exo_model.keras!")
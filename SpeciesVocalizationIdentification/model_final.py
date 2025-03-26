# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:32:41 2024
Last edit on Thu Jan 23 10:17:20 2025
@author: EcoVision Analytics
This program is intended for use exclusively by the Zambiae Carnivore Programme
All rights to use and distribution are solely at the ZCPs discretion
"""

import os
import librosa
import numpy as np
import tensorflow as tf
import scipy.signal
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input, Resizing, Add
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


species_calls_folder = 'species_calls'
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
SR = 22050  # Sample rate
N_MFCC = 40  # n of MFCCs
N_FOLDS = 5  # n of folds for cross-validation
MAX_SAMPLES_PER_CLASS = 1000  # Max samples per class

# These are the functions that implement data augmentation. The function names are pretty self explanatory,
# but let me know if you have any questions in here. 
def add_noise(audio, noise_factor=None):
    if noise_factor is None:
        noise_factor = np.random.uniform(0.003, 0.007)
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def add_background_noise(audio, noise_factor=None):
    if noise_factor is None:
        noise_factor = np.random.uniform(0.1, 0.3)
    background = np.random.randn(len(audio))
    return (audio + noise_factor * background) / (1 + noise_factor)

def apply_random_filter(audio, sr):
    b, a = scipy.signal.butter(N=np.random.randint(1, 4), 
                             Wn=np.random.uniform(0.1, 0.9), 
                             btype='low')
    return scipy.signal.filtfilt(b, a, audio)

def change_pitch(audio, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-3, 3)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def change_speed(audio, speed_factor=None):
    if speed_factor is None:
        speed_factor = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def time_shift(audio, shift_max=None):
    if shift_max is None:
        shift_max = np.random.uniform(0.1, 0.3)
    shift = np.random.randint(int(len(audio) * -shift_max), int(len(audio) * shift_max))
    return np.roll(audio, shift)

def mixup(x1, x2, y1, y2, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

# This is testing model performance under differing noise conditions. 
def test_model_robustness(model, X_test, noise_levels=[0.01, 0.05, 0.1]):
    original_pred = model.predict(X_test)
    results = []
    
    for noise_level in noise_levels:
        noisy_X = X_test + np.random.normal(0, noise_level, X_test.shape)
        noisy_pred = model.predict(noisy_X)
        agreement = np.mean(np.argmax(original_pred, axis=1) == np.argmax(noisy_pred, axis=1))
        results.append((noise_level, agreement))
    
    return results

# This is where we extract features from the audio itself. 
def extract_combined_features_from_audio(audio, sr):
    # MFCC features with more coefficients
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    
    # Mel spectrogram with delta features
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    
    # Additional features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # Spectral contrast 
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    # Combine all features
    combined_features = np.hstack((
        mfccs_mean,  # 40 features
        mfccs_std,   # 40 features
        mel_mean,    # 128 features
        mel_std,     # 128 features
        np.mean(spectral_centroids),  # 1 feature
        np.std(spectral_centroids),   # 1 feature
        np.mean(spectral_rolloff),    # 1 feature
        np.mean(chroma, axis=1),      # 12 features
        np.mean(zcr),                 # 1 feature
        np.std(zcr),                  # 1 feature
        np.mean(bandwidth),           # 1 feature
        np.std(bandwidth),            # 1 feature
        np.mean(contrast, axis=1)     # 7 features
    ))
    
    
    if len(combined_features) < 335:
        padding = np.zeros(335 - len(combined_features))
        combined_features = np.concatenate([combined_features, padding])
    elif len(combined_features) > 335:
        combined_features = combined_features[:335]
    
    return combined_features

# This is weighting classes. Please let me know if you have any questions here. Essentially,
# it is a class balancing issue for the beta training of this model. Once we have some more
# extracted data, we can balance the weighting further.

def calculate_class_weights(y):
    total = len(y)
    classes = np.unique(y)
    class_weights = {}
    for c in classes:
        weight = total / (len(classes) * np.sum(y == c))
        # Boost weights for leopard and elephant
        if label_encoder.classes_[c] in ['leopard', 'elephant']:
            weight *= 1.2
        class_weights[c] = weight
    return class_weights

def load_data():
    labels, features = [], []
    species_folders = ['leopard', 'lion', 'hyena', 'jackal', 'elephant']
    
    for species in species_folders:
        folder_path = os.path.join(species_calls_folder, species)
        files = os.listdir(folder_path)
        np.random.shuffle(files)
        
        # This is increasing samples for specific classes. This was done to improve model performance.
        class_max_samples = MAX_SAMPLES_PER_CLASS
        if species in ['leopard', 'elephant']:
            class_max_samples = int(MAX_SAMPLES_PER_CLASS * 1.2)
        elif species == 'jackal':
            class_max_samples = int(MAX_SAMPLES_PER_CLASS * 0.9)
        
        sample_count = 0
        for file in files:
            if sample_count >= class_max_samples:
                break
                
            if file.endswith((".wav", ".mp3")):
                file_path = os.path.join(folder_path, file)
                audio, sr = librosa.load(file_path, sr=SR)
                
                
                features.append(extract_combined_features_from_audio(audio, sr))
                labels.append(species)
                sample_count += 1
                
                # This is class specific data augementation. This was done to improve model performance.
                if sample_count < class_max_samples:
                    if species in ['leopard', 'elephant']:
                        augmentations = [
                            (add_noise, {'noise_factor': np.random.uniform(0.001, 0.005)}),
                            (change_pitch, {'sr': sr, 'n_steps': np.random.uniform(-2, 2)}),
                            (time_shift, {'shift_max': 0.15}),
                            (add_background_noise, {'noise_factor': np.random.uniform(0.05, 0.15)})
                        ]
                    elif species == 'jackal':
                        augmentations = [
                            (add_noise, {'noise_factor': np.random.uniform(0.004, 0.008)}),
                            (change_pitch, {'sr': sr, 'n_steps': np.random.uniform(-3, 3)}),
                            (change_speed, {'speed_factor': np.random.uniform(0.95, 1.05)})
                        ]
                    else:
                        augmentations = [
                            (add_noise, {}),
                            (change_pitch, {'sr': sr}),
                            (change_speed, {}),
                            (time_shift, {}),
                            (add_background_noise, {}),
                            (apply_random_filter, {'sr': sr})
                        ]
                    
                    for aug_func, aug_args in augmentations:
                        if sample_count >= class_max_samples:
                            break
                        aug_audio = aug_func(audio, **aug_args)
                        features.append(extract_combined_features_from_audio(aug_audio, sr))
                        labels.append(species)
                        sample_count += 1
    
    return np.array(features), np.array(labels)

def create_model(input_shape):
    l2_reg = tf.keras.regularizers.l2(1e-4)
    
    input_tensor = Input(shape=input_shape)
    x = Resizing(158, 64)(input_tensor)

    
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(bn1)
    drop1 = Dropout(0.25)(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(drop1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(bn2)
    bn2 = BatchNormalization()(conv2)
    res2 = Conv2D(128, (1, 1), kernel_regularizer=l2_reg)(drop1)
    add2 = Add()([bn2, res2])
    pool2 = MaxPooling2D((2, 2))(add2)
    drop2 = Dropout(0.35)(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(drop2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg)(bn3)
    bn3 = BatchNormalization()(conv3)
    res3 = Conv2D(256, (1, 1), kernel_regularizer=l2_reg)(drop2)
    add3 = Add()([bn3, res3])
    pool3 = MaxPooling2D((2, 2))(add3)
    drop3 = Dropout(0.4)(pool3)
    flat = Flatten()(drop3)
    dense1 = Dense(512, activation='relu', kernel_regularizer=l2_reg)(flat)
    bn4 = BatchNormalization()(dense1)
    drop4 = Dropout(0.45)(bn4)
    dense2 = Dense(256, activation='relu', kernel_regularizer=l2_reg)(drop4)
    drop5 = Dropout(0.45)(dense2)
    output = Dense(5, activation='softmax')(drop5)

    return Model(inputs=input_tensor, outputs=output)

def cosine_decay_scheduler(epoch, lr):
    initial_lr = 3e-4
    decay_steps = 200
    alpha = 1e-6
    decay = 0.5 * (1 + np.cos(np.pi * epoch / decay_steps))
    return max(initial_lr * decay, alpha)

if __name__ == "__main__":
    
    features, labels = load_data()
    
    # This is reshaping features to 2D before scaling (fix for dimensionality issue)
    features = features.reshape(features.shape[0], -1)
    
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    
    features_scaled = features_scaled.reshape(-1, 335, 1, 1)
    
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    y = to_categorical(y)
    
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    
   
    all_predictions = []
    all_true_labels = []
    
    
    for train_idx, val_idx in kfold.split(features_scaled):
        print(f'\nFold {fold_no}')
        
        X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        
        class_weights = calculate_class_weights(y_train.argmax(axis=1))
        
        
        input_shape = (335, 1, 1)  
        model = create_model(input_shape)
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=3e-4,
                weight_decay=2e-4
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            LearningRateScheduler(cosine_decay_scheduler),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.15,
                patience=12,
                min_lr=1e-6,
                min_delta=1e-4
            ),
            ModelCheckpoint(
                f'best_model_fold_{fold_no}.keras',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=12,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        
        scores = model.evaluate(X_val, y_val, verbose=0)
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        
        
        y_pred = model.predict(X_val)
        all_predictions.extend(np.argmax(y_pred, axis=1))
        all_true_labels.extend(np.argmax(y_val, axis=1))
        
        
        print("\nTesting model robustness:")
        robustness_results = test_model_robustness(model, X_val)
        for noise_level, agreement in robustness_results:
            print(f"Noise level {noise_level:.2f}: {agreement:.4f} prediction stability")
        
        
        plt.figure(figsize=(12, 4))
        
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy - Fold {fold_no}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - Fold {fold_no}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_history_fold_{fold_no}.png')
        plt.close()
        
        fold_no += 1
    
    
    print('\nK-FOLD CROSS VALIDATION RESULTS')
    for i in range(len(acc_per_fold)):
        print(f'Fold {i+1} - Loss: {loss_per_fold[i]:.4f} - Accuracy: {acc_per_fold[i]:.4f}')
    print(f'Average accuracy: {np.mean(acc_per_fold):.4f} (+/- {np.std(acc_per_fold):.4f})')
    
    
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=label_encoder.classes_,
                              digits=4))
    
    
    model.save('final_species_model.keras')
    
    # This is wrapping the model in a TensorFlow Lite container
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    with open('species_model.tflite', 'wb') as f:
        f.write(tflite_model)
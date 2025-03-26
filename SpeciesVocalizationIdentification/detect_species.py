# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:50:19 2024

@author: megal
"""
import os
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
import csv

# Parameters
SR = 22050  # Sample rate
N_MFCC = 40  # Updated number of MFCCs
WINDOW_SIZE = 10 * SR  # Window, 10 seconds
HOP_SIZE = 1 * SR  # Overlap between windows
BUFFER_SIZE = 5 * SR  # Buffer for detection (5 seconds)
MINIMUM_CALL_SEPARATION = 2  # Minimum seconds between calls
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence for a detection
OVERLAP_THRESHOLD = 2.0  # Maximum seconds between calls to be considered the same call
species_list = ['leopard', 'lion', 'hyena', 'jackal', 'elephant']

def extract_features_from_chunk(audio_chunk):
    # MFCC features with more coefficients
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=SR, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    
    # Mel spectrogram with delta features
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=SR, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    
    # Additional features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_chunk, sr=SR)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_chunk, sr=SR)[0]
    chroma = librosa.feature.chroma_stft(y=audio_chunk, sr=SR)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_chunk)[0]
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_chunk, sr=SR)[0]
    
    # Spectral contrast (additional feature)
    contrast = librosa.feature.spectral_contrast(y=audio_chunk, sr=SR)
    
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
    
    # Ensure the feature vector has exactly 335 dimensions
    if len(combined_features) < 335:
        padding = np.zeros(335 - len(combined_features))
        combined_features = np.concatenate([combined_features, padding])
    elif len(combined_features) > 335:
        combined_features = combined_features[:335]
    
    return combined_features

def predict_species_for_chunk(audio_chunk, interpreter, input_details, output_details):
    features = extract_features_from_chunk(audio_chunk)
    features = features.reshape(1, 335, 1, 1).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get confidence scores and predicted class
    confidence_scores = output_data[0]
    predicted_class = np.argmax(confidence_scores)
    confidence = confidence_scores[predicted_class]
    
    # Return -1 if confidence is below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return -1, 0.0
    
    return predicted_class, confidence

def merge_overlapping_calls(detected_calls, overlap_threshold=OVERLAP_THRESHOLD):
    """
    Merge overlapping detections of the same species into single calls.
    overlap_threshold: maximum time gap (in seconds) to consider calls as part of the same vocalization
    """
    if not detected_calls:
        return []
    
    # Sort by start time
    sorted_calls = sorted(detected_calls, key=lambda x: x[0])
    merged_calls = []
    current_call = list(sorted_calls[0])  # Convert to list for mutability
    
    for call in sorted_calls[1:]:
        start, end, species, confidence = call
        
        # If this call overlaps with current call and is same species
        if (start <= current_call[1] + overlap_threshold and 
            species == current_call[2]):
            # Extend current call
            current_call[1] = max(current_call[1], end)
            # Update confidence to maximum confidence
            current_call[3] = max(current_call[3], confidence)
        else:
            # Add current call to merged list and start new current call
            merged_calls.append(tuple(current_call))
            current_call = list(call)
    
    # Add the last call
    merged_calls.append(tuple(current_call))
    
    # Filter out very short detections (less than 1 second)
    merged_calls = [call for call in merged_calls if call[1] - call[0] >= 1.0]
    
    return merged_calls

def process_audio_in_chunks(file_path, interpreter, input_details, output_details):
    audio, sr = librosa.load(file_path, sr=SR)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    detected_calls = []
    chunk_starts = np.arange(0, duration, HOP_SIZE / SR)
    
    print(f"Processing {len(chunk_starts)} chunks...")
    
    for i, start in enumerate(chunk_starts):
        end = start + WINDOW_SIZE / SR
        if end > duration:
            break
        audio_chunk = audio[int(start * SR):int(end * SR)]
        
        predicted_class, confidence = predict_species_for_chunk(audio_chunk, interpreter, input_details, output_details)
        
        if predicted_class != -1:
            detected_calls.append((start, end, predicted_class, confidence))
        
        if i % 100 == 0:  # Print progress every 100 chunks
            print(f"Processed {i}/{len(chunk_starts)} chunks...")
    
    # Merge overlapping detections
    merged_calls = merge_overlapping_calls(detected_calls)
    print(f"Found {len(merged_calls)} unique calls after merging overlapping detections")
    
    return merged_calls

def extract_and_save_calls(file_path, detected_calls, output_folder):
    # Load audio in chunks to handle long files
    with sf.SoundFile(file_path) as audio_file:
        sr = audio_file.samplerate
        
        for i, (start, end, predicted_class, confidence) in enumerate(detected_calls):
            # Calculate positions in samples
            start_samples = int(start * sr)
            end_samples = int(end * sr)
            buffer_samples = int(BUFFER_SIZE)
            
            # Calculate read positions with buffer
            read_start = max(0, start_samples - buffer_samples)
            read_length = (end_samples - start_samples) + (2 * buffer_samples)
            
            # Seek to the correct position and read the chunk
            audio_file.seek(read_start)
            call_segment = audio_file.read(read_length, dtype='float32')
            
            # Skip if segment is too short
            if len(call_segment) < sr:  # Skip if less than 1 second
                continue
            
            # Create species-specific subfolder
            species_folder = os.path.join(output_folder, species_list[predicted_class])
            os.makedirs(species_folder, exist_ok=True)
            
            # Save with species name and timestamp
            filename = f"{species_list[predicted_class]}_call_{int(start)}s.wav"
            save_path = os.path.join(species_folder, filename)
            
            # Save the audio segment
            sf.write(save_path, call_segment, sr)
            
            # Print progress with confidence score
            print(f"Saved call {i+1}/{len(detected_calls)}: {filename} (Confidence: {confidence:.2f})")

def save_detection_report(detected_calls, file_path, output_folder):
    report_data = []
    for start, end, predicted_class, confidence in detected_calls:
        species = species_list[predicted_class]
        report_data.append([
            os.path.basename(file_path),
            species,
            start,
            end,
            f"{confidence:.3f}"
        ])
    
    csv_file = os.path.join(output_folder, 'detection_report.csv')
    
    header = ['File Name', 'Species Detected', 'Start Time (s)', 'End Time (s)', 'Confidence']
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  
            writer.writerow(header)
        writer.writerows(report_data)

def run_detection_on_all_files(input_folder, model_path, output_folder):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of all audio files
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp3', '.wav'))]
    total_files = len(audio_files)
    
    print(f"Found {total_files} audio files to process")
    print(f"Using confidence threshold of {CONFIDENCE_THRESHOLD}")
    
    for file_num, file_name in enumerate(audio_files, 1):
        file_path = os.path.join(input_folder, file_name)
        
        print(f"\nProcessing file {file_num}/{total_files}: {file_name}")
        print("=" * 50)
        
        detected_calls = process_audio_in_chunks(file_path, interpreter, input_details, output_details)
        
        if detected_calls:
            print(f"Extracting and saving {len(detected_calls)} detected calls...")
            extract_and_save_calls(file_path, detected_calls, output_folder)
            save_detection_report(detected_calls, file_path, output_folder)
            print(f"Finished processing {file_name}")
        else:
            print(f"No calls detected in {file_name}")

def main():
    input_folder = 'Input'  # Folder with audio files to process
    model_path = 'species_model.tflite'  # Path to the TFLite model
    output_folder = 'Output'  # Folder to save detected calls and reports

    print("Starting species detection process...")
    run_detection_on_all_files(input_folder, model_path, output_folder)
    print("\nProcessing complete!")

if __name__ == '__main__':
    main()
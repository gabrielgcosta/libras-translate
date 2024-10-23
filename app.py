import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image
import os
import csv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

# Inicializando o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Caminho para o arquivo CSV onde salvaremos os landmarks
csv_file = 'libras_dataset.csv'
image_folder = 'images'

# Função para salvar os landmarks em um CSV
def save_landmarks_to_csv(hand_landmarks, label):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [label]
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        writer.writerow(row)

# Função para processar a imagem e capturar landmarks
def process_image_for_landmarks(image, label):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Converter a imagem para RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Verificar se foi encontrada alguma mão
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                save_landmarks_to_csv(hand_landmarks, label)
            return True
        else:
            return False

# Função para extrair o label do nome do arquivo
def extract_label_from_filename(filename):
    return filename.split('-')[0]  # Pega a parte antes do '-' como label

# Função para processar as imagens da pasta e gerar o CSV
@app.route('/process_images', methods=['POST'])
def process_images():
    if not os.path.exists(image_folder):
        return jsonify({'error': 'Images folder not found'}), 400

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return jsonify({'error': 'No images found in the folder'}), 400

    # Verificar se o arquivo CSV já existe, caso contrário, criar com cabeçalhos
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['label'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)

    processed_images = []

    for image_file in image_files:
        # Extrair o label do nome do arquivo
        label = extract_label_from_filename(image_file)
        image_path = os.path.join(image_folder, image_file)
        
        # Abrir a imagem usando OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            success = process_image_for_landmarks(image, label)
            if success:
                processed_images.append(image_file)
            else:
                print(f"Falha ao processar landmarks para {image_file}")

    if processed_images:
        return jsonify({'message': f'Processed images: {processed_images}'}), 200
    else:
        return jsonify({'error': 'No valid images were processed'}), 400

# Função para treinar o modelo RandomForest
def train_model():
    global clf
    if not os.path.exists(csv_file):
        return 'No data available to train the model', 400

    data = pd.read_csv(csv_file)

    # Separar características e rótulos
    X = data.drop('label', axis=1)
    y = data['label']

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

# Rota para treinar o modelo após capturar os dados
@app.route('/train', methods=['POST'])
def train():
    train_model()
    return jsonify({'message': 'Model trained successfully!'})

# Função que gera o vídeo da câmera e realiza a predição em tempo real
def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Converte a imagem para RGB para o MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Desenha as marcações nas mãos e faz a predição do gesto
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Prever gesto
                    gesture = predict_gesture(hand_landmarks)
                    cv2.putText(frame, f'Traducao: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Codificar o frame para enviar como stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Envia o frame como uma resposta de stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Função para prever gestos em tempo real
def predict_gesture(hand_landmarks):
    row = []
    for landmark in hand_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z])

    if clf:
        # Criar um DataFrame temporário para corresponder às colunas do modelo
        columns = [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        df = pd.DataFrame([row], columns=columns)
        
        prediction = clf.predict(df)
        return prediction[0]
    else:
        return "Model not trained yet"

# Rota para exibir a página HTML com o stream de vídeo
@app.route('/translate')
def translate():
    return render_template('translate.html')

# Rota para fornecer o stream de vídeo para a página
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

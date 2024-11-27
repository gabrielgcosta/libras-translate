import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image
import os
import csv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask import jsonify

app = Flask(__name__)

# Inicializando o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Caminho para o arquivo CSV onde salvaremos os landmarks
csv_file = 'libras_dataset.csv'
image_folder = 'images'
test_folder = 'test'
resized_image_folder = 'resized_images'
image_size = (128, 128)  # Tamanho desejado para as imagens redimensionadas

# Função para redimensionar as imagens antes do processamento
def resize_images(input_dir, output_dir, size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        output_class_path = os.path.join(output_dir, class_dir)

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    img_resized = img.resize(size, Image.LANCZOS)  # Substitui ANTIALIAS por LANCZOS
                    new_file_path = os.path.join(output_class_path, img_file)
                    img_resized.save(new_file_path, "PNG")
                    print(f"Resized and saved image: {new_file_path}")
            except Exception as e:
                print(f"Error resizing image {img_file}: {e}")

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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                save_landmarks_to_csv(hand_landmarks, label)
            return True
        else:
            return False

# Função para extrair o label do nome do arquivo
def extract_label_from_filename(filename):
    return filename.split('-')[0]

# Função para processar as imagens redimensionadas e gerar o CSV
@app.route('/process_images', methods=['POST'])
def process_images():
    # Redimensionar imagens, se necessário, e salvar na pasta de destino
    if not os.path.exists(resized_image_folder):
        resize_images(image_folder, resized_image_folder, image_size)

    # Coletar imagens de todas as subpastas de 'resized_image_folder'
    image_files = []
    for subfolder in os.listdir(resized_image_folder):
        subfolder_path = os.path.join(resized_image_folder, subfolder)
        subfolder_path = 'resized_images\A'
        if os.path.isdir(subfolder_path):  # Verificar se é uma subpasta
            for img_file in os.listdir(subfolder_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(subfolder_path, img_file))

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
        # Extrair o label do nome da subpasta
        label = os.path.basename(os.path.dirname(image_file))
        
        # Abrir a imagem usando OpenCV
        image = cv2.imread(image_file)
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
    
# Rota para validar o modelo usando as imagens da pasta test
@app.route('/validate_model', methods=['POST'])
def validate_model():
    if not os.path.exists(test_folder):
        return jsonify({'error': 'Test folder not found'}), 400

    # Lista de resultados que será usada para gerar o CSV
    validation_results = []  # Renomeando a variável para evitar conflito
    
    # Itera sobre cada pasta (cada letra) na pasta de teste
    for letter_folder in os.listdir(test_folder):
        letter_path = os.path.join(test_folder, letter_folder)
        
        # Verifica se é um diretório
        if os.path.isdir(letter_path):
            total_images = 0
            correct_predictions = 0
            incorrect_predictions = 0
            
            # Itera sobre cada imagem na subpasta
            for image_file in os.listdir(letter_path):
                image_path = os.path.join(letter_path, image_file)
                
                # Abre a imagem e processa com o modelo
                image = cv2.imread(image_path)
                if image is not None:
                    total_images += 1
                    
                    # Extrai landmarks da imagem
                    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        hand_results = hands.process(image_rgb)  # Renomeado para evitar conflito

                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                predicted_letter = predict_gesture(hand_landmarks)
                                
                                # Verifica se a predição está correta
                                if predicted_letter == letter_folder:
                                    correct_predictions += 1
                                else:
                                    incorrect_predictions += 1

            # Adiciona os resultados dessa pasta à lista de resultados
            validation_results.append({
                'Letter': letter_folder,
                'Total Images': total_images,
                'Correct Predictions': correct_predictions,
                'Incorrect Predictions': incorrect_predictions
            })

    # Gera o arquivo CSV com os resultados
    csv_filename = 'validation_results.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Letter', 'Total Images', 'Correct Predictions', 'Incorrect Predictions']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in validation_results:
            writer.writerow(result)

    return jsonify({'message': f'Validation complete. Results saved to {csv_filename}'}), 200

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

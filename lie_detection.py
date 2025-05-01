import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from deepface import DeepFace
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Configuration ---
VIDEO_SOURCE = 0  # Webcam (0) or video file path
FRAME_RATE = 5    # Optimized FPS for better performance on CPU
MICROEXPRESSION_DURATION = 0.5  # Seconds (microexpressions are ~0.1-0.5s)
HISTORY_FRAMES = int(FRAME_RATE * MICROEXPRESSION_DURATION)  # Frames for analysis
DECEPTION_THRESHOLD = 0.6  # Confidence for deception flag
IMAGE_SIZE = (64, 64)  # Optimized size for performance
FRAME_RESOLUTION = (640, 480)  # Reduced resolution for faster processing
MAX_FRAMES = 300  # Stop after 300 frames (60 seconds at 5 FPS)

# --- Deep Learning Model (CNN + RNN for Temporal Analysis) ---
class MicroexpressionModel(nn.Module):
    def __init__(self):
        super(MicroexpressionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.gru = nn.GRU(128, 64, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(64, 7)  # 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc_deception = nn.Linear(7, 1)
        nn.init.xavier_uniform_(self.fc_deception.weight)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_out = x.view(batch_size * seq_len, c, h, w)
        c_out = F.relu(self.conv1(c_out))
        c_out = self.pool(c_out)
        c_out = F.relu(self.conv2(c_out))
        c_out = self.pool(c_out)
        c_out = c_out.view(batch_size * seq_len, -1)
        c_out = F.relu(self.fc1(c_out))
        c_out = c_out.view(batch_size, seq_len, -1)
        g_out, _ = self.gru(c_out)
        g_out = g_out[:, -1, :]
        emotion_out = self.fc2(g_out)
        deception_out = torch.sigmoid(self.fc_deception(emotion_out))
        return emotion_out, deception_out

# --- Face Detection and Emotion Analysis with DeepFace (OpenCV Backend) ---
def detect_faces_deepface(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv',
                                enforce_detection=False, silent=True)
        if result and isinstance(result, list) and len(result) > 0:
            face = result[0]
            region = face['region']
            emotion = face['emotion']
            print(f"Emotion probabilities: {emotion}")
            dominant_emotion = max(emotion, key=emotion.get)
            if sum(emotion.values()) / len(emotion) < 0.2:
                dominant_emotion = 'neutral'
            FaceRect = namedtuple('FaceRect', ['left', 'top', 'right', 'bottom', 'emotion'])
            return [FaceRect(
                left=region['x'],
                top=region['y'],
                right=region['x'] + region['w'],
                bottom=region['y'] + region['h'],
                emotion=dominant_emotion
            )]
    except Exception as e:
        print(f"DeepFace error: {e}")
        return []
    return []

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
])

# --- Main Processing Loop ---
def process_video():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_RESOLUTION[1])
    frame_buffer = deque(maxlen=HISTORY_FRAMES)
    model = MicroexpressionModel()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')) if os.path.exists('model.pth') else model.state_dict())
    model.eval()
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    frame_count = 0
    deception_scores = []  # Store all deception scores

    print("Press 'q' with the video window active to stop the script.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video source ended.")
            break

        faces = detect_faces_deepface(frame.copy())

        for face in faces:
            x, y = face.left, face.top
            w, h = face.right - x, face.bottom - y
            face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

            face_img = cv2.resize(face_img, IMAGE_SIZE)
            face_tensor = transform(face_img).unsqueeze(0)
            frame_buffer.append(face_tensor)

            if len(frame_buffer) == HISTORY_FRAMES:
                sequence = torch.stack(list(frame_buffer), dim=1)
                with torch.no_grad():
                    emotions, deception_prob = model(sequence)

                emotion_probs = F.softmax(emotions, dim=1).numpy()[0]
                dominant_emotion = face.emotion
                if emotion_labels[np.argmax(emotion_probs)] != dominant_emotion and sum(emotion_probs) / 7 < 0.3:
                    dominant_emotion = 'neutral'
                deception_score = deception_prob.item()
                deception_scores.append(deception_score)  # Store the score

                smoothed_score = gaussian_filter1d([deception_score], sigma=2)[0]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Deception: {smoothed_score:.2f}", (x, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if smoothed_score > DECEPTION_THRESHOLD:
                    cv2.putText(frame, "Possible Deception!", (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if frame_count % 30 == 0:
                    cv2.imwrite(f"output/frame_{frame_count}.png", frame)
                    plt.figure(figsize=(8, 6))
                    plt.bar(emotion_labels, emotion_probs, color='skyblue')
                    plt.title('Emotion Probabilities')
                    plt.xlabel('Emotion')
                    plt.ylabel('Probability')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'output/emotion_plot_{frame_count}.png')
                    plt.close()

        cv2.imshow('Microexpression Lie Detection', frame)
        frame_count += 1

        if frame_count >= MAX_FRAMES:
            print(f"Reached maximum frames ({MAX_FRAMES}). Stopping.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q'. Stopping.")
            break

    # Calculate average deception score and make final decision
    if deception_scores:
        avg_deception_score = sum(deception_scores) / len(deception_scores)
        print(f"Average Deception Score: {avg_deception_score:.2f}")
        if avg_deception_score < DECEPTION_THRESHOLD:
            print("Final Decision: The person is NOT LYING (average score below threshold).")
        else:
            print("Final Decision: The person MIGHT BE LYING (average score above threshold).")
    else:
        print("No deception scores recorded. Ensure faces are detected during the run.")

    cap.release()
    cv2.destroyAllWindows()
    torch.save(model.state_dict(), 'model.pth')

# --- Run the System ---
if __name__ == '__main__':
    if not os.path.exists("output"):
        os.makedirs("output")
    process_video()
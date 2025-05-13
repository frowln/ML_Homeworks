import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import os
import time

# Инициализация решений MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Инициализация переменных
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
)

# Инициализация классификатора лиц OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

YOUR_NAME = "Sayid"
YOUR_SURNAME = "Salizhanov"

# Создаем папку для эталонных фотографий, если её нет
if not os.path.exists('reference_photos'):
    os.makedirs('reference_photos')


def count_existing_photos():
    """Подсчет существующих фотографий в папке"""
    count = 0
    for file in os.listdir('reference_photos'):
        if file.endswith(('.jpg', '.png')):
            count += 1
    return count


def count_fingers(hand_landmarks):
    """Подсчет поднятых пальцев на основе точек руки"""
    finger_tips = [8, 12, 16, 20]  # Индексы кончиков пальцев (кроме большого)
    finger_base = [6, 10, 14, 18]  # Индексы оснований пальцев

    raised_fingers = 0

    # Проверяем большой палец отдельно
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        raised_fingers += 1

    # Проверяем остальные пальцы
    for tip, base in zip(finger_tips, finger_base):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            raised_fingers += 1

    return raised_fingers


def detect_emotion(frame, face_location):
    """Определение эмоции с помощью DeepFace"""
    try:
        x, y, w, h = face_location
        face_img = frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "Unknown"


def load_reference_faces():
    """Загрузка эталонных лиц"""
    reference_faces = []

    for filename in os.listdir('reference_photos'):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join('reference_photos', filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Обнаружение лиц на изображении
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y + h, x:x + w]
                # Изменяем размер для стандартизации
                face_roi = cv2.resize(face_roi, (100, 100))
                reference_faces.append(face_roi)

    return reference_faces


def is_my_face(frame, face_location, reference_faces):
    """Проверка, является ли лицо лицом пользователя"""
    try:
        x, y, w, h = face_location
        face_img = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))

        # Сравниваем с эталонными лицами
        for ref_face in reference_faces:
            # Вычисляем разницу между лицами
            diff = cv2.absdiff(gray, ref_face)
            similarity = 1 - (np.sum(diff) / (100 * 100 * 255))

            # Если сходство больше 0.8, считаем что это лицо пользователя
            if similarity > 0.8:
                return True

        return False
    except:
        return False


def collect_reference_photos():
    """Сбор эталонных фотографий с веб-камеры"""
    cap = cv2.VideoCapture(0)
    photo_count = count_existing_photos()

    print("\nРежим сбора фотографий:")
    print("1. Нажмите 'c' чтобы сделать фото")
    print("2. Нажмите 'q' чтобы закончить сбор фотографий")
    print("Рекомендуется иметь 3-5 фотографий с разных ракурсов\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Конвертируем в RGB для MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка лиц
        face_results = face_detection.process(frame_rgb)

        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)

                # Рисуем прямоугольник вокруг лица
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Показываем количество сделанных фотографий
        cv2.putText(frame, f"Photos taken: {photo_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Показываем инструкции
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Photo Collection Mode', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if face_results.detections:
                # Сохраняем только область с лицом
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    # Добавляем отступы вокруг лица
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    width = min(w - x, width + 2 * padding)
                    height = min(h - y, height + 2 * padding)

                    face_img = frame[y:y + height, x:x + width]
                    photo_count += 1
                    cv2.imwrite(f'reference_photos/face_{photo_count}.jpg', face_img)
                    print(f"Фото {photo_count} сохранено")
                    time.sleep(0.5)  # Небольшая задержка между фотографиями
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return photo_count


# Основная программа
def main():
    # Сначала собираем фотографии
    photo_count = collect_reference_photos()

    if photo_count == 0:
        print("Не было сделано ни одной фотографии. Программа завершается.")
        return

    print(f"\nВсего фотографий: {photo_count}. Загрузка эталонных лиц...")

    # Загружаем эталонные лица
    reference_faces = load_reference_faces()

    if not reference_faces:
        print("Не удалось загрузить эталонные лица. Программа завершается.")
        return

    print("Запуск основной программы...")
    print("Нажмите 'q' для выхода из программы\n")

    # Инициализация веб-камеры
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Конвертируем в RGB для MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка рук
        hand_results = hands.process(frame_rgb)
        finger_count = 0

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_landmarks)

        # Обработка лиц
        face_results = face_detection.process(frame_rgb)

        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)

                # Рисуем прямоугольник вокруг лица
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Проверяем, является ли лицо лицом пользователя
                if is_my_face(frame, (x, y, width, height), reference_faces):
                    # Добавляем текст в зависимости от количества пальцев только для лица пользователя
                    if finger_count == 1:
                        text = YOUR_NAME
                    elif finger_count == 2:
                        text = YOUR_SURNAME
                    elif finger_count == 3:
                        emotion = detect_emotion(frame, (x, y, width, height))
                        text = f"Emotion: {emotion}"
                    else:
                        text = "Unknown"
                else:
                    text = "Unknown"

                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Отображаем количество пальцев
        cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Показываем кадр
        cv2.imshow('Face and hand recognition', frame)

        # Прерываем цикл при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np
import math
import re  # For sanitizing mathematical expressions
from pynput.keyboard import Controller
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import pygame

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

keyboard = Controller()

pygame.mixer.init()
click_sound = pygame.mixer.Sound("C:\\Users\\YUVRAJ SINGH\\virtual keyboard\\click-234708.mp3")  # Update path if needed

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

text = ""
current_language = "English"
gesture_hold_time = 2
last_gesture_time = 0  # Time to track the hold duration
caps_lock_on = False  # Caps lock state

# Define button class
class Button:
    def __init__(self, pos, text, size=[70, 70]):
        self.pos = pos
        self.size = size
        self.text = text

# Define key layouts for English and Symbols keyboards
keys = {
    "English": [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "SAVE"],
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "CL"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "CAPS"],
        ["Z", "X", "C", "V", "B", "SP", "N", "M", ",", ".", "/", "TOG"]
    ],
    "Symbols": [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "SAVE"],
        ["+", "-", "*", "/", "=", "%", "^", "(", ")", "[", "]", "CL"],
        ["{", "}", "|", "\\", ":", ";", "'", "\"", "<", ">", "SP"],
        ["@", "#", "$", "&", "_", "~", "`", "!", "?", ".", ",", "TOG"]
    ]
}

# Function to draw buttons with Unicode support
def drawAll(img, buttonList, font_path="C:/Windows/Fonts/arial.ttf"):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 40)  # Adjust font size as needed

    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (96, 96, 96), cv2.FILLED)
        draw.text((x + 15, y + 15), button.text, font=font, fill=(255, 255, 255, 255))

    return np.array(img_pil)

# Function to calculate distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Evaluate mathematical expression
def evaluate_expression(expression):
    try:
        safe_expression = re.sub(r'[^\d\+\-\*/\(\)\.\s]', '', expression)  # Remove unsafe characters
        return str(eval(safe_expression))
    except:
        return "Error"

# Initialize button list
buttonList = []
def update_buttons(language):
    global buttonList
    buttonList = []
    
    button_width = 80
    button_height = 80
    num_columns = max(len(row) for row in keys[language])
    num_rows = len(keys[language])

    total_keyboard_width = button_width * num_columns
    total_keyboard_height = button_height * num_rows
    screen_width, screen_height = 1280, 720
    start_x = (screen_width - total_keyboard_width) // 2 - 100
    start_y = (screen_height - total_keyboard_height) // 2

    for i, row in enumerate(keys[language]):
        for j, key in enumerate(row):
            buttonList.append(Button([start_x + j * button_width, start_y + i * button_height], key))

update_buttons(current_language)

def save_text():
    global text
    with open("typed_text.txt", "w", encoding="utf-8") as file:
        file.write(text)
    print("Text saved to 'typed_text.txt'")

# Main loop
while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    landmarks = []
    frame = drawAll(frame, buttonList, font_path="C:/Windows/Fonts/arial.ttf")

    if results.multi_hand_landmarks:
        for hn in results.multi_hand_landmarks:
            for id, lm in enumerate(hn.landmark):
                hl, wl, cl = frame.shape
                cx, cy = int(lm.x * wl), int(lm.y * hl)
                landmarks.append([id, cx, cy])

    if landmarks:
        try:
            x8, y8 = landmarks[8][1], landmarks[8][2]
            x4, y4 = landmarks[4][1], landmarks[4][2]
            dis = calculate_distance(x8, y8, x4, y4)

            if dis > 50:
                cv2.circle(frame, (x8, y8), 20, (0, 255, 0), cv2.FILLED)
                if last_gesture_time == 0:
                    last_gesture_time = cv2.getTickCount()

                elapsed_time = (cv2.getTickCount() - last_gesture_time) / cv2.getTickFrequency()
                if elapsed_time > gesture_hold_time:
                    for button in buttonList:
                        xb, yb = button.pos
                        wb, hb = button.size
                        if xb < x8 < xb + wb and yb < y8 < yb + hb:
                            k = button.text
                            if k == "SP":
                                text += ' '
                            elif k == "CL":
                                text = text[:-1]
                            elif k == "TOG":
                                current_language = "Symbols" if current_language == "English" else "English"
                                update_buttons(current_language)
                            elif k == "CAPS":
                                caps_lock_on = not caps_lock_on
                            elif k == "SAVE":
                                save_text()
                            else:
                                if caps_lock_on:
                                    text += k.upper()
                                else:
                                    text += k.lower()
                                if text.endswith("="):  # Check if input ends with "="
                                    expression = text[:-1]  # Remove "="
                                    result = evaluate_expression(expression)
                                    text += result
                            last_gesture_time = 0
                            click_sound.play()
            else:
                last_gesture_time = 0
        except:
            pass

    # Display typed text
    cv2.rectangle(frame, (20, 600), (1260, 680), (255, 255, 255), cv2.FILLED)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
    draw.text((30, 620), text, font=font, fill=(0, 0, 0, 255))
    frame = np.array(img_pil)
    
    cv2.imshow('Virtual Keyboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
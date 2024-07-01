import RPi.GPIO as GPIO
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
from skimage.transform import resize

# Pin numbers are set in BCM mode
GPIO.setmode(GPIO.BCM)

# Suppressing unnecessary warnings on the console
GPIO.setwarnings(False)

# 7-Segment pin numbers
GPIO.setup(7, GPIO.OUT)  # A
GPIO.setup(8, GPIO.OUT)  # B
GPIO.setup(25, GPIO.OUT)  # C
GPIO.setup(20, GPIO.OUT)  # D
GPIO.setup(21, GPIO.OUT)  # E
GPIO.setup(12, GPIO.OUT)  # F
GPIO.setup(16, GPIO.OUT)  # G

# Array holding the LEDs to light up according to the detected emotion
# Anger -> A, Disgust -> D, Fear -> F, Happiness -> H, Neutrality -> N, Sadness -> S, Surprised -> P in hexadecimal
emotion_leds = [0x77, 0x3D, 0x47, 0x37, 0x15, 0x5B, 0x67]

print('Loading ..')

model = tflite.Interpreter("model.tflite")  # loading the trained model
model.allocate_tensors()
ipt = model.get_input_details()[0]
opt = model.get_output_details()[0]

print('Load Success')

# Cascade model required for face detection
cascade_path = "haarcascade_frontalface_default.xml"
cascade_face = cv2.CascadeClassifier(cascade_path)

def capture_face(img_, x, y, w, h):  # captures the face region from the given image
    return img_[y:y + h, x:x + w]

def preprocess_image(raw):  # prepares the captured face image for the model
    img = resize(raw, (200, 200, 3))
    img = np.expand_dims(img, axis=0)
    if np.max(img) > 1:
        img = img / 255.0
    return img

def set_emotion(raw, x, y, w, h):
    emotion_lbl = ''
    img = capture_face(raw, x, y, w, h)
    img = preprocess_image(img)
    model.set_tensor(ipt['index'], img.astype(np.float32))
    model.invoke()
    res = model.get_tensor(opt['index'])
    classes = np.argmax(res, axis=1)
    if classes == 0:
        emotion_lbl = 'Anger'
    elif classes == 1:
        emotion_lbl = 'Disgust'
    elif classes == 2:
        emotion_lbl = 'Fear'
    elif classes == 3:
        emotion_lbl = "Happiness"
    elif classes == 4:
        emotion_lbl = "Neutrality"
    elif classes == 5:
        emotion_lbl = 'Sadness'
    else:
        emotion_lbl = 'Surprise'

    display_letter(classes)
    return emotion_lbl

def display_letter(class_no):  # sends the 7-segment pin numbers to light up according to the detected emotion
    if class_no > 6:  # protects against errors for class_no greater than 6 and negative values
        class_no = 6
    elif class_no < 0:
        class_no = 0

    pin = emotion_leds[class_no]
    seven_segment_port(pin)

def seven_segment_port(pin_no):  # lights up the appropriate LEDs on the 7-segment display according to the detected emotion
    if pin_no & 0x01 == 0x01:  # check for the 0th bit
        GPIO.output(7, GPIO.HIGH)
    else:
        GPIO.output(7, GPIO.LOW)
    if pin_no & 0x02 == 0x02:  # check for the 1st bit
        GPIO.output(8, GPIO.HIGH)
    else:
        GPIO.output(8, GPIO.LOW)
    if pin_no & 0x04 == 0x04:  # check for the 2nd bit
        GPIO.output(25, GPIO.HIGH)
    else:
        GPIO.output(25, GPIO.LOW)
    if pin_no & 0x08 == 0x08:  # check for the 3rd bit
        GPIO.output(20, GPIO.HIGH)
    else:
        GPIO.output(20, GPIO.LOW)
    if pin_no & 0x10 == 0x10:  # check for the 4th bit
        GPIO.output(21, GPIO.HIGH)
    else:
        GPIO.output(21, GPIO.LOW)
    if pin_no & 0x20 == 0x20:  # check for the 5th bit
        GPIO.output(12, GPIO.HIGH)
    else:
        GPIO.output(12, GPIO.LOW)
    if pin_no & 0x40 == 0x40:  # check for the 6th bit
        GPIO.output(16, GPIO.HIGH)
    else:
        GPIO.output(16, GPIO.LOW)

vid = cv2.VideoCapture(0) # getting the image from the webcam
predicted_emotion = 'Neutrality'
img = np.zeros((200, 200, 3))

while True:
    # capturing an image from the webcam
    ret, frame = vid.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade_face.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150)
    )

    emotion_label = ''
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        predicted_emotion = set_emotion(gray_img, x, y, w, h)

    # displaying the image to the user
    cv2.imshow('Frame', frame)
    # program ends when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing the video and closing windows
vid.release()
cv2.destroyAllWindows()
GPIO.cleanup() 

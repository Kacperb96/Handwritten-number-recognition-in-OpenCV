import numpy as np
import cv2
import pickle

# Utworzenie modelu kamery
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Załadowanie modelu treningowego
saved_model = open("model.p","rb")
model = pickle.load(saved_model)

# Załadowanie obrazów
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = camera.read()
    if not success:
        cv2.waitKey(20)
        print('Nie odczytano kamery')
        continue
    img = np.array(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    # Przewidywania
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if probVal > 0.65:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
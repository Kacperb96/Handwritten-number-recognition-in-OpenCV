import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D,MaxPooling2D, Flatten, Dropout, Dense

# Importowanie danych z folderów
count = 0
images = []
classNumber = []
myList = os.listdir('Dane')
print("Liczba klas:",len(myList))
numberOfClasses = len(myList)
print("Importowanie klas")
for x in range (0,numberOfClasses):
    myPicList = os.listdir('Dane'+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread('Dane'+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNumber.append(x)
    print(x,end= " ")
print(" ")
print("Całkowita liczba danych = ",len(images))

# Zamiana na tablice z numpy
images = np.array(images)  # TODO Upewnić się
classNumber = np.array(classNumber)

# Podzielenie danych
X_train,X_test,y_train,y_test = train_test_split(images,classNumber,test_size=0.2)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2)

# Eksploracja danych
print('Liczba danych treningowych:', X_train.shape[0])
print('Liczba danych testowych:', X_test.shape[0])
print('Liczba danych walidacji: ', X_validation.shape[0])
print('Rozmiar obrazka:', X_train.shape[1],'x',X_train.shape[2])

# Przygotowanie do wyswietlenia w openCV
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train= np.array(list(map(preProcessing,X_train)))
X_test= np.array(list(map(preProcessing,X_test)))
X_validation= np.array(list(map(preProcessing,X_validation)))


# Przygotowanie danych
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

y_train = to_categorical(y_train,numberOfClasses)
y_test = to_categorical(y_test,numberOfClasses)
y_validation = to_categorical(y_validation, numberOfClasses)

# Budowanie modelu
def myModel():
    #model = Sequential()
    #model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(units=128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(numberOfClasses, activation='softmax'))
    #model.summary()
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2)
    noOfNodes= 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=((32, 32, 3)[0],
                           (32, 32, 3)[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()

# Trenowanie modelu
history = model.fit(X_train, y_train, batch_size=128, epochs=10,validation_data=(X_test, y_test))

# Ocena modelu
score = model.evaluate(X_test, y_test)
print('Strata:', score[0])
print('Dokładność:', score[1])

# Wykresy
def make_loss_plot(history):
    plt.figure(1)
    plt.title('Strata trenowania')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend(['Trening','Walidacja'])
    plt.show()

def make_accuracy_plot(history):
    plt.figure(2)
    plt.title('Dokładność trenowania')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend(['Trening', 'Walidacja'])
    plt.show()

make_accuracy_plot(history)
make_loss_plot(history)

#### Zapis modelu
pickle_out= open("model.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()
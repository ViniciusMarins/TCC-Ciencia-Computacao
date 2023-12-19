import os
import random
import keyboard
import time

# tratamento de imagens
from PIL import Image
import cv2
import numpy as np
import dlib

# plots de imagens
import matplotlib.pyplot as plt

# rede neural e afins
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

folder_path = 'Base_dados'

data = {}
classes = []

images = []
labels = []
le = LabelEncoder()

imgs_train, imgs_test, labels_train, labels_test = None,None,None,None

def varrer_base_dados():
    global images, labels, le, imgs_train, imgs_test, labels_train, labels_test, classes, data

    data.clear()
    classes.clear()
    images = []
    labels = []

    for pasta_name in os.listdir(folder_path):
        print(pasta_name)
        name, prontuario = pasta_name.split('-', 1)
        path = folder_path + '/' + pasta_name

        for filename in os.listdir(path):
            imagem = Image.open(path + '/' + filename)
            imagem_array = np.array(imagem)

            if prontuario not in data:
                data[prontuario] = {'nome': name, 'imagens': []}
            data[prontuario]['imagens'].append(imagem_array)

            # imagem_array = np.expand_dims(imagem_array, axis=0)

            # imgAug = ImageDataGenerator(
            #                             horizontal_flip=True,
            #                             vertical_flip=True)
            
            # imgGen = imgAug.flow(imagem_array)
            
            # counter = 0

            # for (i, newImage) in enumerate(imgGen):
            #     counter += 1

            #     newImage = np.squeeze(newImage, axis=0)
            #     data[prontuario]['imagens'].append(newImage)
            #     # ao gerar 2 imagens, parar o loop
            #     if counter == 5:
            #         break

    for prontuario, info in data.items():
        classes.append(info['nome'])
        for img in info['imagens']:
            # Conversão em escala de cinza
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images.append(img_gray)
            labels.append(info['nome'])

    images = np.array(images)
    labels = np.array(labels)

    classes = sorted(classes)

    # Separação em dados de teste e treinamento
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.15)

    # Normalização dos dados
    imgs_train = imgs_train / 255
    imgs_test = imgs_test / 255

    # Encoder das labels
    labels_train = le.fit_transform(labels_train)
    labels_test = le.transform(labels_test)


historico = None


def treinar_rede_neural():

    global historico

    model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape=(100,100,1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2),),
    keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),),
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(
            filepath="image_classifier.keras",
            save_best_only=True,
            monitor="val_loss",
        )
    ]

    history = model.fit(
        imgs_train,
        labels_train,
        epochs=15,
        validation_data = (imgs_test,labels_test),
        callbacks=callbacks,
    )

    print("Alunos:")
    print(classes)
    
    val_loss, val_acc = model.evaluate(imgs_test, labels_test)
    print('Loss do teste:', val_loss)
    print('Acurácia do teste:', val_acc)
    
    historico = history

def gerar_graficos(history):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, "r", label="Treino acc")
    plt.plot(epochs, val_accuracy, "b", label="Val acc")
    plt.xlabel("Épocas")
    plt.ylabel("%s")
    plt.title("Acurácia de Treino e Validação")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "r", label="Treino loss")
    plt.plot(epochs, val_loss, "b", label="Val loss")
    plt.xlabel("Épocas")
    plt.ylabel("%s")
    plt.title("Loss de Treino e Validação")
    plt.legend()
    plt.show()

def testar_base_teste():
    model = keras.models.load_model('image_classifier.keras')

    index = random.choice(range(len(imgs_test)))
    prec_img = imgs_test[index]

    # showSingleImageTeste(prec_img,"Imagem Selecionada", (5,5))
    # cv2.imshow(le.inverse_transform([labels_test[index]]).item(),prec_img)
    # cv2.waitKey(0)

    prec_img = np.array([prec_img])
    #print(prec_img)

    prediction = model.predict(prec_img)
    index = np.argmax(prediction)

    array = []
    for arr in prediction:
        for valor in arr:
            array.append(round(valor*100,5))

    for i in range(len(classes)):
        print(f'Porcentagem {classes[i]} : {array[i]}%')
    
    print(classes)
    print(f'\nPredição de classe é {classes[index]}')

    showSingleImageTeste(prec_img,f"Predição: {classes[index]} \n Porcentagem: {array[index]}%" , (5,5))
    plt.show()

def testar_fotos_selecionadas(caminho):
    model = keras.models.load_model('image_classifier.keras')

    hog_face_detector = dlib.get_frontal_face_detector()

    prec_img = cv2.imread(caminho)
    prec_img = cv2.cvtColor(prec_img, cv2.COLOR_BGR2RGB)

    faces = hog_face_detector(prec_img)
    face_img_resized = None

    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()

        # Recorta a região da face da imagem original
        face_img = prec_img[y1:y2, x1:x2]
        face_img_resized = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_LANCZOS4)
        

        new_face_img_resized = cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2GRAY)

        prediction = model.predict(np.array([new_face_img_resized]) / 255)
        index = np.argmax(prediction)

        array = []
        for arr in prediction:
            for valor in arr:
                array.append(round(valor * 100, 5))

        for i in range(len(classes)):
            print(f'Porcentagem {classes[i]} : {array[i]}%')

        print(f'\nPredição de classe é {classes[index]}')
        showSingleImage(face_img_resized,f"Predição: {classes[index]} \n Porcentagem: {array[index]}%" , (5,5))
        showSingleImage(prec_img,"Foto Selecionada",(5,5))
          
def reconhecimento_Presencas(progressBar):
    model = keras.models.load_model('image_classifier.keras')
    progressBar.show()
    progressBar.setValue(25)

    captura = cv2.VideoCapture(0)

    progressBar.setValue(60)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    progressBar.setValue(100)
    progressBar.hide()

    hog_face_detector = dlib.get_frontal_face_detector()
    alunos_detectados = set()
    
    while True:
        ret, frame = captura.read()

        frame = cv2.flip(frame, 1)

        cv2.imshow('Captura de imagem', frame)

        if keyboard.is_pressed('e'):
            print('Captura realizada.')
            prec_img = frame
            break

        key = cv2.waitKey(1)
        if key == 27 or key == 81 or key == 113:  
            break
    
    captura.release()
    cv2.destroyAllWindows()
    time.sleep(0.1)
    # prec_img = cv2.imread(caminho)
    prec_img = cv2.cvtColor(prec_img, cv2.COLOR_BGR2RGB)

    faces = hog_face_detector(prec_img)
    face_img_resized = None

    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()

        # Recorta a região da face da imagem original
        face_img = prec_img[y1:y2, x1:x2]
        face_img_resized = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_LANCZOS4)
        
        new_face_img_resized = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY)

        prediction = model.predict(np.array([new_face_img_resized]) / 255)
        index = np.argmax(prediction)

        array = []
        for arr in prediction:
            for valor in arr:
                array.append(round(valor * 100, 5))

        for i in range(len(classes)):
            print(f'Porcentagem {classes[i]} : {array[i]}%')

        print(f'\nPredição de classe é {classes[index]}')
        alunos_detectados.add(classes[index])
        showSingleImage(face_img_resized,f"Predição: {classes[index]} \n Porcentagem: {array[index]}%" , (5,5))
        showSingleImage(prec_img,"Foto Selecionada",(5,5))
    
    return alunos_detectados

def showSingleImage(img, title, size):
    fig, axis = plt.subplots(figsize = size)

    axis.imshow(img, 'gray')
    axis.set_title(title, fontdict = {'fontsize': 17, 'fontweight': 'medium'})
    plt.show()

def showSingleImageTeste(img, title, size):
    fig, axis = plt.subplots(figsize=size)

    axis.imshow(img.squeeze(), cmap='gray')  # usa squeeze() para remover a dimensão de comprimento 1
    axis.set_title(title, fontdict={'fontsize': 17, 'fontweight': 'medium'})
    #plt.show()
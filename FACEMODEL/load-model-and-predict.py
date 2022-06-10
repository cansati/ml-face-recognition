import numpy as np
import keras
import matplotlib.pyplot as plt
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#Modele verilecek eğitim görsellerinin dosya yolu
train_dir = "./face-recognition-dataset/Original Images/Original Images/"

#Görsellerden Veri Setleri oluşturmak için kullandığımız preprocessing sınıfı
generator = ImageDataGenerator()

#12. Satırda belirlediğimiz dosya yolundaki görsellerden veri seti oluşturuyoruz
train_ds = generator.flow_from_directory(train_dir,target_size=(224, 224),batch_size=32)
classes = list(train_ds.class_indices.keys())

model = keras.models.load_model('./model.h5')

def predict_image(image_path):

    #görsel dosya yolundan yüklenir
    img = image.load_img(image_path, target_size=(224,224,3))

    #ekranda gösterilir
    plt.imshow(img)
    plt.show()

    #görsel numpy dizisine dönüştürülür
    x = image.img_to_array(img)

    #görselin [x, y] boyutu, modele sunulmak üzere [x, y, 1] boyutuna genişletilir
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    #verilen görsel tahmin edilir
    pred = model.predict(images, batch_size=32)

    #doğru olan ve tahmin edilen loglara yazdırılır
    print("Actual: "+(image_path.split("/")[-1]).split("_")[0])
    print("Predicted: "+classes[np.argmax(pred)])


predict_image("./face-recognition-dataset/Original Images/Original Images/Brad Pitt/Brad Pitt_102.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Charlize Theron/Charlize Theron_26.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Henry Cavill/Henry Cavill_28.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Tom Cruise/Tom Cruise_27.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Robert Downey Jr/Robert Downey Jr_106.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Natalie Portman/Natalie Portman_25.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Lisa Kudrow/Lisa Kudrow_34.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Ellen Degeneres/Ellen Degeneres_20.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Dwayne Johnson/Dwayne Johnson_29.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Elizabeth Olsen/Elizabeth Olsen_11.jpg")
predict_image("./face-recognition-dataset/Original Images/Original Images/Ramitan Ishdavlatov/Ramitan Ishdavlatov_11.jpg")

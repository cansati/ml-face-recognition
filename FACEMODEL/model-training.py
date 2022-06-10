import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import matplotlib.pyplot as plt
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#Modele verilecek eğitim görsellerinin dosya yolu
train_dir = "./face-recognition-dataset/Original Images/Original Images/"

#Görsellerden Veri Setleri oluşturmak için kullandığımız preprocessing sınıfı
generator = ImageDataGenerator()

#12. Satırda belirlediğimiz dosya yolundaki görsellerden veri seti oluşturuyoruz
train_ds = generator.flow_from_directory(train_dir,target_size=(224, 224),batch_size=32)
classes = list(train_ds.class_indices.keys())

#Sıralı cins modelimizi ilk çalıştırma için tanımlıyoruz
model = Sequential()

#Modelimize aşağıdaki parametrelerle ilk katmanımızı ekliyoruz
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))

#Görselimizi maxpooling yöntemiyle 1/4 oranında küçülterek ve bilgi kaybını minimize ederek eğitimimizi hızlandırmayı hedefliyoruz
model.add(MaxPooling2D(pool_size=(2,2)))

#Verilerimizin değerini 0-255'ten 0-1 aralığına getiriyoruz ki, değeri büyük olan veriler, küçük olanları ezip önemsizleştirmesin
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#Bir üstteki katman ile sonraki katman arasındaki ağırlık bağlantılarını modelin ezberlemesini önlemek için
# rastgele olarak %20 oranında kopartıyoruz
model.add(Dropout(0.2))

#2D olan görselimizi tek boyutlu bir numpy dizisine dönüştürüyoruz
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))

#Son katmanımızın nöron sayısı, elimizdeki kategori sayısı kadardır. Her bir nöron, bir kategoriyi temsil eder.
#Verilen görsel için tahminde bulunulduğunda bu son nöronlardan birisi yanar.
model.add(Dense(len(classes),activation='softmax'))

#model aşağıdaki parametrelerle derlenir ve eğitime hazır hale getirilir.
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ["accuracy"])

#modelin parametrelerinin özeti gösterilir
model.summary()

#Model, veri seti ile epoch sayısı ve her seferinde kaç görselin modele sunulacağı belirlenerek eğitime başlar
history = model.fit(train_ds,epochs= 30, batch_size=32)

model.save("model.h5")

#matplotlib kütüphanesi aracılığıyla doğruluk ve kayıp miktarı zamana bağlı olarak çizdirilir.
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.xlabel('Time')
plt.legend(['accuracy', 'loss'])
plt.show()

#!wget http://cb.lk/covid_19
#!unzip covid_19

train_path = "CovidDataset/Train"
val_path = "CovidDataset/Test"

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image




#cnn based model in keras

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])




#train from scratch

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True
)

test_dataset = image.ImageDataGenerator(rescale=1./255)




train_generator = train_datagen.flow_from_directory(
    'CovidDataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
train_generator.class_indices





validation_generator =  test_dataset.flow_from_directory(
    'CovidDataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
    



hist = model.fit(
    train_generator,
    steps_per_epoch = 5,
    epochs =10,
    validation_data = validation_generator,
    validation_steps = 2
)





model.save("model_adv.h5")




model.evaluate(train_generator)




model.evaluate(validation_generator)



model=load_model("model_adv.h5")





import os
import numpy as np
train_generator.class_indices
y_actual=[]
y_test = []




for i in os.listdir("./CovidDataset/Val/Normal/"):
  img = image.load_img("./CovidDataset/Val/Normal/"+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict(img)
  y_test.append(p[0,0])
  y_actual.append(1)

for i in os.listdir("./CovidDataset/Val/Covid/"):
  img = image.load_img("./CovidDataset/Val/Covid/"+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  #p = np.array(model.predict(img))
  #p = np.argmax(model.predict(img),axis=-1)
  p = model.predict(img)
  y_test.append(p[0,0])
  y_actual.append(0)
  
  
  
  
y_actual = np.array(y_actual)
y_test = np.array(y_test)

print(y_actual)
print(y_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_actual,y_test)
sns.heatmap(cm,cmap="plasma",annot = True)

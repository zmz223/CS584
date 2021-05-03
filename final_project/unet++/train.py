import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from unetplusplus import Xnet

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
 # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model, save_model, Sequential
import numpy as np
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, save_model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.keras.backend.tensorflow_backend import set_session




# In[2]:


import os
import numpy as np       # linear algebra
import pandas as pd      # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.util import montage
from skimage.io   import imread


print(os.getcwd())
# # Locate all the image and mask files


IMG_PATH = "/labs/colab/CS584-Spring2021/Lungs-CT/2d_images/"
MASK_PATH = "/labs/colab/CS584-Spring2021/Lungs-CT/2d_masks/"
DS_FACT = 2
SEED=42

all_image_files = []
all_mask_files  = []

for file in os.listdir(IMG_PATH):
    all_image_files.append(IMG_PATH+file)

for file in os.listdir(MASK_PATH):
    all_mask_files.append(MASK_PATH+file)



print('No. of images:', len(all_image_files))
print(all_image_files[0])
print(all_mask_files[0])

images = []
for img in all_image_files:
    temp = np.expand_dims(imread(img)/255., -1)
    gray_image_3band = np.repeat(temp, repeats = 3, axis = -1)
    images.append(gray_image_3band)

#masks = []
#for mask in all_mask_files:
#    temp = np.expand_dims(imread(mask)/255., -1)
 #   gray_image_3band = np.repeat(temp, repeats = 3, axis = -1)
  #  masks.append(gray_image_3band)

images = np.array(images)
#masks = np.array(masks)


#images   = np.stack((np.expand_dims(imread(i)/255., -1) for i in all_image_files),0)
masks   = np.stack((np.expand_dims(imread(i)/255., -1) for i in all_mask_files),0)





X_train, X_test, y_train,  y_test = train_test_split(images, masks, test_size=0.1)

print('X_train - len/shape:', len(X_train), X_train.shape)
print('Y_train is {}, min is {}, max is {}, mean is {}'.format(y_train.shape, y_train.min(), y_train.max(), y_train.mean()))
print('X_test  - len/shape:', len(X_test), y_test.shape)
print(images.shape[:])


model = Xnet(encoder_weights='imagenet', decoder_block_type='transpose')
model.summary()
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# In[68]:



callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-lung.hdf5', verbose=1, save_best_only=True)
]


# In[69]:

history = model.fit(X_train, y_train, batch_size=10, epochs=50, callbacks=callbacks,validation_data=(X_test, y_test))


# The validation loss is 0.0267

# # Evaluation

# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


import pickle
with open('predictions', 'wb') as file_pi:
        pickle.dump(y_pred, file_pi)
with open('hist', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

with open('X_test', 'wb') as file_pi:
        pickle.dump(X_test, file_pi)
# In[ ]:
with open('y_test', 'wb') as file_pi:
        pickle.dump(y_test, file_pi)





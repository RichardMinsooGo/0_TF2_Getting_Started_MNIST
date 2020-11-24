import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cifar10 = tf.keras.datasets.cifar10
IMG_SIZE = 32

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

print(X_train.shape[0])
print(X_test.shape[0])
# sys.exit()

# cv2.resize(X_train[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])

# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, verbose=1)

model.evaluate(X_test,  Y_test, verbose=2)              

"""
verbose default is 1

verbose=0 (silent)

verbose=1 (progress bar)

Train on 186219 samples, validate on 20691 samples
Epoch 1/2
186219/186219 [==============================] - 85s 455us/step - loss: 0.5815 - acc: 
0.7728 - val_loss: 0.4917 - val_acc: 0.8029
Train on 186219 samples, validate on 20691 samples
Epoch 2/2
186219/186219 [==============================] - 84s 451us/step - loss: 0.4921 - acc: 
0.8071 - val_loss: 0.4617 - val_acc: 0.8168


verbose=2 (one line per epoch)

Train on 186219 samples, validate on 20691 samples
Epoch 1/1
 - 88s - loss: 0.5746 - acc: 0.7753 - val_loss: 0.4816 - val_acc: 0.8075
Train on 186219 samples, validate on 20691 samples
Epoch 1/1
 - 88s - loss: 0.4880 - acc: 0.8076 - val_loss: 0.5199 - val_acc: 0.8046
 
 """
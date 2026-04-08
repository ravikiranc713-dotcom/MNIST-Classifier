import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

x_train  = x_train/255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)

model = keras.Sequential([layers.Dense(128,activation='relu',input_shape=(784,)),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(10, activation='softmax')
                          ])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train,y_train,epochs=5,batch_size=32)

test_loss,test_acc = model.evaluate(x_test,y_test)
print("Test Accuracy:",test_acc)

train_preds = model.predict(x_train)
train_pred_labels = np.argmax(train_preds, axis=1)

plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {train_pred_labels[i]} | True: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

test_preds = model.predict(x_test)
test_pred_labels = np.argmax(test_preds,axis=1)

wrong_indices = np.where(test_pred_labels!=y_test)[0]

plt.figure(figsize=(8,8))

for j in range(9):
    idx = wrong_indices[j]
    plt.subplot(3,3,j+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {test_pred_labels[idx]} | True: {y_test[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

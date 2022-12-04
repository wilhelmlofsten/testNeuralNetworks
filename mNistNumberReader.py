import tensorflow as tf
import numpy as np



model = tf.keras.models.Sequential([            # 3 st layers,128, 64 och 10 är antal neurons /layer

    
    tf.keras.layers.Dense(128, activation= "relu"),
    tf.keras.layers.Dense(64, activation= "relu"),
    tf.keras.layers.Dense(10, activation= "softmax"),

])


#ladda MNIST
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

# återskapar, formar och konverterar inputen till float32

x_train = (x_train / 255.).reshape([-1,784]).astype(np.float32)
x_test = (x_test / 255.).reshape([-1,784]).astype(np.float32)

#convert labebls till one hot vectors

y_train = tf.one_hot(y_train,10)
y_test = tf.one_hot(y_test,10)


#fixar själva träningen

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.shuffle(500).batch(32)

model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(train_data)


#kollar accuary av lästa siffran

def accuracy(y_pred,y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred,-1), tf.argmax(y_true,-1))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32), axis = -1)


pred = model(x_test)
print(f"Test accuracy of read number:  {accuracy(pred, y_test)}")     #bör ha ~96% accuracy

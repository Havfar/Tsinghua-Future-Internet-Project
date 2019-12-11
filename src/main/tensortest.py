# imports
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# Load Cifar10 training data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#plt.imshow(x_train[0])
print(x_train[0:10].shape)

# Normalize the data (go from values between 0 to 255 to values between 0 and 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build model
model = tf.keras.applications.VGG19(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(32,32,3),
    pooling=None,
    classes=10
)

# optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


# Compile model
model.compile(optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train (fit) model
model.fit(x_train[0:1000], y_train[0:1000], epochs=10)

# evaluate the training

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

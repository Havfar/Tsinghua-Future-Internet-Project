import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

import resnet

NUM_CPUS = 1
NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 60
USING_CPU = True

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


# Normalization --> gives zero main 
def preprocess(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


# Changes images for better results of ML
def augmentation(x, y):
    # Draws black border around image
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    
    # Picks a random square out of the image (dimension: HEIGHT, WEIGHT)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    # Flips the image randomly (50% chance)
    x = tf.image.random_flip_left_right(x)
    return x, y	


# A function used in the callback of the model.fit part of the code
# This is used to change the learning rate during learning.
def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

# Loads data to x, labels to y
(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Split data in train and test set
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# ?? 
tf.random.set_seed(22)

# Creates a new datasetobject of the augmented images, but retains the original order
# Shuffle the training set
# Normalize it (preprocess)
train_dataset = train_dataset.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

# Batch the data. Drop remainder = modulo the training data on batch size
# Think its just used for multi-GPU on single machine.
# We might be able to use this for multi-CPU distributed though. Not sure
test_dataset = test_dataset.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

input_shape = (32, 32, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


# Compiling keras models
if USING_CPU:
    if NUM_CPUS == 1:
      # ???
      model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
      # Actual compiling, define optimizer, loss function, metric for evaluation
      model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])
    else:
      # Create a streategy for tensorflow
      mirrored_strategy = tf.distribute.MirroredStrategy()
      with mirrored_strategy.scope():
        model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)

          # Actual compiling, define optimizer, loss function, metric for evaluation
        model.compile(
                  optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])  
else:
  if NUM_GPUS == 1:

      #???
      model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)

      # Actual compiling, define optimizer, loss function, metric for evaluation
      model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])
  else:
      # Create a streategy for tensorflow
      mirrored_strategy = tf.distribute.MirroredStrategy()
      with mirrored_strategy.scope():
        model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)

          # Actual compiling, define optimizer, loss function, metric for evaluation
        model.compile(
                  optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])  

# Logging
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

# TensorBoard is imported, used to log data. Good for visualizing the performance and logging
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

# Imported LearningRateScheduler is passed the schedule function
# Allows us to invoke callback functions during training
lr_schedule_callback = LearningRateScheduler(schedule)

# Train model, takes tensorflow objects (test_dataset, train_dataset)
# These could be numpy arrays, but using tf objects allow us to transform the data
model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=1,
          # Callback function is added, the tensorboard for logging, and lr_schedule to change the
          # learning rate during machine learning
          callbacks=[tensorboard_callback, lr_schedule_callback])

# Evaluate the trained model
model.evaluate(test_dataset)

model.save('model.h5')

new_model = keras.models.load_model('model.h5')
 
new_model.evaluate(test_dataset)
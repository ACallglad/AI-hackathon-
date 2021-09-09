

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image

train_dir = "training" 
test_dir = "test"
batch_size = 128
img_shape = (64, 64, 3)
epochs=10
num_classes = len(os.listdir(train_dir))
idx_to_name = os.listdir(train_dir)
name_to_idx = dict([(v, k) for k, v in enumerate(idx_to_name)])

def data_to_df(data_dir, subset=None, train_size=None):
    ''' Creating DataFrame for loading Filename and Label
    
    Args:
        Data_dir: 
            Data_dir Path
        Subset: 
            - train- for spliting data in train and val
    
     '''
    df = pd.DataFrame(columns=['filenames', 'labels'])

    filenames = []
    labels = []
    for dataset in os.listdir(data_dir):
        img_list = os.listdir(os.path.join(data_dir, dataset))

        label = name_to_idx[dataset]

        for image in img_list:
            filenames.append(os.path.join(data_dir, dataset, image))
            labels.append(label)

    df["filenames"] = filenames
    df["labels"] = labels
    
    if subset == "train":
        train_df, val_df = train_test_split(df, train_size=train_size, shuffle=True)    
        return train_df, val_df
    return df

train_df, val_df = data_to_df(train_dir, subset="train", train_size=0.8)

class CustomDataGenerator(tf.keras.utils.Sequence):

    ''' Custom DataGenerator to load img 
    
    Arguments:
        data_frame = pandas data frame in filenames and labels format
        batch_size = divide data in batches
        shuffle = shuffle data before loading
        img_shape = image shape in (h, w, d) format
        augmentation = data augmentation to make model rebust to overfitting
    
    Output:
        Img: numpy array of image
        label : output label for image
    '''
    
    def __init__(self, data_frame, batch_size=10, img_shape=None, augmentation=True, num_classes=None):
        self.data_frame = data_frame
        self.train_len = self.data_frame.shape[0]
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_classes = num_classes
        print(f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        self.data_frame = shuffle(self.data_frame)
        return int(self.train_len/self.batch_size)

    def on_epoch_end(self):
        # fix on epoch end it's not working, adding shuffle in len for alternative
        pass
    
    def __data_augmentation(self, img):
        img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.3)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img
        
    def __get_image(self, file_id):
        img = np.asarray(Image.open(file_id))
        img = np.resize(img, self.img_shape)
        #img = self.__data_augmentation(img)
        img = preprocess_input(img)

        return img

    def __get_label(self, label_id):
        return label_id

    def __getitem__(self, idx):
        batch_x = self.data_frame["filenames"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.data_frame["labels"][idx * self.batch_size:(idx + 1) * self.batch_size]
        # read your data here using the batch lists, batch_x and batch_y
        x = [self.__get_image(file_id) for file_id in batch_x] 
        y = [self.__get_label(label_id) for label_id in batch_y]

        return np.array(x), np.array(y)

train_data = CustomDataGenerator(train_df, batch_size=batch_size, img_shape=img_shape, num_classes=num_classes)
val_data = CustomDataGenerator(val_df, batch_size=batch_size, img_shape=img_shape, num_classes=num_classes)

base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=img_shape)
base_model.trainable= True

class BuildModel(tf.keras.Model):
    def __init__(self, base_model):
        super(BuildModel, self).__init__()
        self.base_model = base_model
        self.globalaveragepooling = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(2)
        
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.globalaveragepooling(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

model = BuildModel(base_model)
model.build(input_shape=(None, 64, 64, 3))

model.summary()

optimizer = optimizers.Adam(learning_rate=1e-5)
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metrics = metrics.SparseCategoricalAccuracy()
val_acc_metrics = metrics.SparseCategoricalAccuracy()

checkpoint_dir = "tmp/"
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metrics.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metrics.update_state(y, val_logits)

epochs = 5
import time
for epoch in range(epochs):
    print(f"Epoch : {epoch}/{epochs}")
    start_time = time.perf_counter()
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        loss_value = train_step(x_batch_train, y_batch_train)
        
        if (step % 20) == 0: 
            print(f"Step: {step} - Training loss : {loss_value}")
            print(f"Seen so far: {(step + 1) * batch_size} samples") 

    train_acc = train_acc_metrics.result()
    print(f"Training Accuracy: {float(train_acc)}")
    train_acc_metrics.reset_states()

    for x_batch_val, y_batch_val in val_data:
        test_step(x_batch_val, y_batch_val)
    
    val_acc = val_acc_metrics.result()
    print(f"Validation Accuracy: {float(val_acc)}")
    val_acc_metrics.reset_states()
    
    total_time = time.perf_counter() - start_time
    
    print(f"Time Taken: {total_time} seconds")
    checkpoint.save(checkpoint_dir)
    print("-"*80)

test_df = data_to_df("test2")
test_data = CustomDataGenerator(train_df, batch_size=64, img_shape=img_shape, num_classes=num_classes)

model.compile(optimizer, loss_fn)

def model_evalution(test_data):
    """ function to test the loss and accuracy on validation data """
    for X_test, y_test in val_data:
        y_pred = model(x_test, training=False)
        val_acc_metrics.update_state(y_test, y_pred)
        accuracy = val_acc_metrics.result()
    
    return float(accuracy)

model_evalution(val_data)

# Creating Json File to submit the solution
final_output = {}
for folder in os.listdir("test2"):
    for image_name in os.listdir(os.path.join("test2", folder)):
        img_ = image.load_img(os.path.join("test2", folder, image_name), 
                              target_size=(64, 64), color_mode="rgb")
        img_arr = image.img_to_array(img_)
        img_arr = preprocess_input(img_arr)
        img_batch = np.array([img_arr])
        output = model.predict(img_batch)
        if output > 0.5:
            final_output[image_name] = 1
        else:
            final_output[image_name] = 0

with open("result.json", "w") as file:
    json.dump(final_output, file)


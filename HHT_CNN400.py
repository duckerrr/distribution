# from make_datasets import make_datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import numpy as np
import os


def plot_acc(history, epochs, val_freq):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['sparse_categorical_accuracy']
    val_accuracy = history.history['val_sparse_categorical_accuracy']
    # val_loss = history.history['val_mean_squared_logarithmic_error']
    # val_mean_squared_logarithmic_error = history.history['val_mean_squared_logarithmic_error']
    fig, ax = plt.subplots(2, 1, sharex="col")
    plt.subplots_adjust(hspace=0.5)
    ax[0].plot(loss, label='loss')
    ax[0].plot(np.arange(0, epochs, val_freq), val_loss, label='val_loss')
    ax[0].set_ylim(np.min(loss)*0.6, np.max(loss)*1.2)
    ax[0].set_title('training and validation loss')
    ax[0].legend()

    ax[1].plot(accuracy, label='accuracy')
    ax[1].plot(np.arange(0, epochs, val_freq), val_accuracy, label='val_accuracy')
    ax[1].set_ylim(np.min(accuracy)*0.9, 1.05)
    ax[1].set_title('training and validation accuracy')
    ax[1].legend()
    ax[1].set_xlabel("epochs")
    plt.show()

class CNN_model(Model):
    def __init__(self, n_class):
        super(CNN_model, self).__init__()
        self.n_class = n_class
        self.c1 = layers.Conv1D(filters=6, kernel_size=3, padding='same',)
        self.p1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')

        self.c2 = layers.Conv1D(filters=12, kernel_size=5, padding='valid')
        self.p2 = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')

        self.f = layers.Flatten()
        self.fc1 = layers.Dense(432, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.last = layers.Dense(self.n_class)
        # self.f2 = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.f(x)
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.last(x)
        return y


if __name__ == '__main__':
    npzfile = np.load('../datasets/datasets_HHT400.npz')
    model_weight_path = r"../model/hht_CNN400/weight.ckpt"
    datasets = npzfile["datasets"][:, :, 3:-2]
    labels = npzfile["labels"].reshape(-1, 1).astype(np.int64)
    validation_freq = 1
    batchs = 64
    epoch = 100
    n_class = 11
    counts = np.bincount(labels[:, 0])
    class_weight = {0:1/counts[0], 1:1/counts[1], 2:1/counts[2], 3:1/counts[3]}
    x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, stratify=labels, random_state=1000)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batchs).prefetch(tf.data.experimental.AUTOTUNE)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(batchs).prefetch(tf.data.experimental.AUTOTUNE)
    model = CNN_model(n_class)
    save_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_weight_path, save_weights_only=True,
                                                     save_best_only=True, mode="max",
                                                     monitor="val_sparse_categorical_accuracy")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.0001,
                                                      patience=5, mode='max',
                                                      baseline=0.01, restore_best_weights=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
                  , metrics=['sparse_categorical_accuracy']
                  # , metrics=['accuracy']
                  , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                  # , loss=tf.keras.losses.MeanAbsoluteError()

                  )
    if os.path.exists(model_weight_path + '.index'):
        print("-" * 8, "load the model", '-' * 8)
        model.load_weights(model_weight_path)
    history = model.fit(train_db, epochs=epoch, validation_data=test_db, validation_freq=validation_freq
                        # , class_weight=class_weight
                        # , callbacks=[early_stopping, save_model]
                        )
    # model_400.build(input_shape=(None, 400, 6))
    model.summary()
    true_epochs = len(history.epoch)
    plot_acc(history, epochs=true_epochs, val_freq=validation_freq)
    # print(history.history['val_sparse_categorical_accuracy'][-1])
    print("最大准确率", np.max(history.history['val_sparse_categorical_accuracy']))
    print("平均准确率:", np.mean(history.history['val_sparse_categorical_accuracy'][-5:]))







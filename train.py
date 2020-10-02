from __future__ import print_function
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras import backend as K
import efficientnet.keras as efn
import numpy
import emnist
# import matplotlib.pyplot as plt


def under_sampling(x, y):
    unique, counts = numpy.unique(y, return_counts=True)
    sampler = dict(zip(unique, counts))
    min_count = min(counts)
    x_new = y_new = None
    for k,v in sampler.items():
        idx = numpy.where(y == k)[0]
        if x_new is None:
            x_new = x[idx][:min_count]
            y_new = y[idx][:min_count]
        else:
            x_new = numpy.concatenate([x_new, x[idx][:min_count]])
            y_new = numpy.concatenate([y_new, y[idx][:min_count]])
    return (x_new, y_new)


lr = 1e-3
batch_size = 128
num_classes = 10
epochs = 100
emnist_mat = 'emnist/emnist-digits.mat'
model_name = 'CNN'  # MLP, CNN, MobileNet, EfficientNet
dataset = 'MNIST+EMNIST'  # MNIST, EMNIST, MNIST+EMNIST
mode = 'train'  # train, test


if dataset == 'MNIST':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = under_sampling(x_train, y_train)
    x_test, y_test = under_sampling(x_test, y_test)
elif dataset == 'EMNIST':
    training_data = emnist.load_data(emnist_mat)
    (x_train, y_train), (x_test, y_test), mapping, num_classes = training_data
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = under_sampling(x_train, y_train)
    x_test, y_test = under_sampling(x_test, y_test)
    training_data = emnist.load_data(emnist_mat)
    (x_train_, y_train_), (x_test_, y_test_), mapping, num_classes = training_data
    x_train = numpy.concatenate([x_train, x_train_])
    y_train = numpy.concatenate([y_train, y_train_])
    x_test = numpy.concatenate([x_test, x_test_])
    y_test = numpy.concatenate([y_test, y_test_])


# plt.figure(figsize=(10,10))
# for i in range(10):
#     data = [(x,t) for x, t in zip(x_train, y_train) if t == i]
#     x, y = data[0]
#     plt.subplot(5,2, i+1)
#     plt.title("len={}".format(len(data)))
#     plt.axis("off")
#     plt.imshow(x, cmap='gray')
# plt.tight_layout()
# plt.show()


if model_name == 'MLP':
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    if model_name != 'CNN':
        x_train = numpy.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), 'constant');
        x_test = numpy.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), 'constant');


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


if model_name == 'MLP':
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
elif model_name == 'CNN':
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
elif model_name == 'MobileNet':
    base_model = keras.applications.mobilenet.MobileNet(
            include_top=True, weights=None,
            input_shape=(32,32,1), classes=num_classes)
    model = Model(inputs=base_model.input, outputs=base_model.output)
elif model_name == 'EfficientNet':
    base_model = efn.EfficientNetB0(
            include_top=True, weights=None,
            input_shape=(32,32,1), classes=num_classes)
    model = Model(inputs=base_model.input, outputs=base_model.output)


if mode == 'train':
    early_stopping = keras.callbacks.EarlyStopping(patience=0, verbose=1)
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=lr),
            metrics=['accuracy'])
    model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs, verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping])
    model.save('model.h5')
else:
    model = load_model('model.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=lr),
            metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


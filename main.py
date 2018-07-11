import os
import numpy as np
from skimage import io, transform
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers import MaxPooling2D, Flatten, Dropout

class CNN:
    def __init__(self, epoch, path, label_path, size, fraction):
        self.epochs = epoch
        self.path = path
        self.label_path = label_path
        self.img_size = size
        self.data_fraction = fraction
        self.data = []
        self.labels = []
        self.test_data = []
        self.train_data = []
        self.test_labels = []
        self.train_labels = []
        return

    def transform_image(self, image):
        return transform.resize(image, (self.img_size, self.img_size, image.shape[2]))

    def LoadData(self):
        images = os.listdir(self.path)

        for image in images:
            if image[-4:] == "jpeg":
                transformed_image = self.transform_image(io.imread(self.path + '/' + image))
                self.data.append(transformed_image)

                label_file = image[:-5] + ".txt"
                with open(self.label_path + '/' + label_file) as f:
                    content = f.readlines()
                    label = int(float(content[0]))
                    l = [0, 0]
                    l[label] = 1
                    self.labels.append(l)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        return

    def test_train_split(self):
        train_data_size = int(len(self.data) * self.data_fraction)
        self.train_data, self.train_labels = self.data[:train_data_size], self.labels[:train_data_size]
        self.test_data, self.test_labels = self.data[train_data_size:], self.labels[train_data_size:]
        return

    def BuildModel(self):
        model = Sequential()
        model.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(self.img_size, self.img_size, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
        model.add(Convolution2D(16, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
        model.add(Flatten())
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return

    def Main(self):
        self.LoadData()
        self.test_train_split()

        print("Train data size = ", len(self.train_data))
        print("Test data size = ", len(self.test_data))

        idx = np.random.permutation(self.train_data.shape[0])
        self.BuildModel()
        self.model.fit(self.train_data[idx], self.train_labels[idx], nb_epoch=self.epochs)

        preds = np.argmax(self.model.predict(self.test_data), axis=1)
        test_labels = np.argmax(self.test_labels, axis=1)

        print(accuracy_score(test_labels, preds))
        return

if __name__ == "__main__":
    cnn = CNN(100, "Data", "Labels", 50, 0.8)
    cnn.Main()
import matplotlib.pyplot as plt
import cv2

def CACDplot(dataset):
    plt.figure(figsize=(10, 5))
    for i in range(10): # El CACD no es correcte al 100%
        plt.subplot(2, 5, i+1)
        img, age = dataset.__getitem__(i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
        plt.title(age+14)
    plt.show()


def generate_Xy_dataset(self):
    X = []
    y = []
    for index in range(self.__len__()):
        img, age = self.__getitem__(index)
        if img is not None:
            X.append(img)
            y.append(age)
        if(index % 10000 == 0):
            print('Img index:', index)

    return X, y
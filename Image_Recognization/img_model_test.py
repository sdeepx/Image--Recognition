from PIL import Image
import numpy as np
from keras.models import load_model
from sys import exit

# labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# model = load_model("train_image_model.h5")
#
# img_path = input("Enter img: ")
# img = Image.open(img_path)
# img = img.resize((32, 32), resample=Image.LANCZOS)
# img_array = np.array(img)
# img_array = img_array.astype("float32")
# img_array /= 255
# img_array = img_array.reshape((1, 32, 32, 3))
# answer = model.predict(img_array)
# img.show()
# print(labels[np.argmax(answer)])


def images(img_path):
    labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    model = load_model("train_image_model.h5")
    img = Image.open(img_path)
    img = img.resize((32, 32), resample=Image.LANCZOS)
    img_array = np.array(img)
    img_array = img_array.astype("float32")
    img_array /= 255
    img_array = img_array.reshape((1, 32, 32, 3))
    answer = model.predict(img_array)
    img.show()
    print(labels[np.argmax(answer)])


def main():
    try:
        img_path = input("Enter img: ")
        images(img_path)
    except (FileNotFoundError, AttributeError, KeyboardInterrupt):
        if FileNotFoundError:
            print("Please Enter your right path...")
            return main()
        if AttributeError:
            print("Please Enter your path name...")
            return main()
        if KeyboardInterrupt:
            print("Process Canceled")
            exit()


if __name__ == '__main__':
    main()

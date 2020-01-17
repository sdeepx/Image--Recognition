from PIL import Image
from keras.datasets import cifar10
from matplotlib import pyplot as plt

# img_path_name = input("Enter Img Path: ")
# img = Image.open(img_path_name)
# img.show()

(X_train, y_train), (X_text, y_text) = cifar10.load_data()

index = int(input("index: "))

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

display_img = X_train[index]
display_label = y_train[index][0]

img_split = Image.fromarray(display_img)
red, green, blue = img_split.split()

plt.imshow(display_img)
plt.show()

print(labels[display_label])

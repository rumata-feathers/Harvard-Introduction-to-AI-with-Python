import cv2
import numpy as np
import os
import sys
import tensorflow as tf
## uncomment to visualize
# import matplotlib.pyplot as plt
import visualkeras

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    ## uncomment to visualize
    # # calculate plot
    # # create new figure for every sign
    # # plt.figure(ind)
    # # every sign can be split to groups of 30 images
    # img_cols = 9
    # img_rows = 5
    # # create grid for showing pics
    # fig, axx = plt.subplots(img_rows, img_cols, figsize=(9, 5))
    # plt.subplots_adjust(left=0.002, bottom=0.002, right=0.998, top=0.998, wspace=0.002, hspace=0.002)
    # for i, j in itertools.product(range(img_rows), range(img_cols)):
    #     axx[i, j].axis('off')

    # iterate though all indexes of the signs
    for ind in range(NUM_CATEGORIES):

        print("Try to load " + str(ind) + " folder in " + data_dir)

        # folder with images of the sign
        folder = os.path.join(data_dir, str(ind))

        ## uncomment to visualize
        # # calculate index
        # index_w, index_h = ind % img_cols, ind // img_cols
        # preview_shown = False

        # iterate through all images in the folder
        for fileName in os.listdir(folder):

            # read image as np array
            image = cv2.imread(os.path.join(folder, fileName))

            # if read append the image
            if image is not None:
                # resize the image
                resized_image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

                ## uncomment to visualize
                # # show the image
                # if not preview_shown:
                #     axx[index_h, index_w].imshow(resized_image, interpolation='nearest')
                #     preview_shown = True

                # append resized image
                images.append(resized_image)

                # append the sign index as label
                labels.append(ind)
    ## uncomment to visualize
    # plt.show()

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.Sequential([
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 64)
        ),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer. Learn 64 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", input_shape=(IMG_WIDTH / 2, IMG_HEIGHT / 2, 128)
        ),
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", input_shape=(IMG_WIDTH / 2, IMG_HEIGHT / 2, 128)
        ),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer. Learn 64 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", input_shape=(IMG_WIDTH / 4, IMG_HEIGHT / 4, 256)
        ),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.2),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(512, activation="relu"),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    visualkeras.layered_view(model, legend=True).show()
    return model


if __name__ == "__main__":
    main()

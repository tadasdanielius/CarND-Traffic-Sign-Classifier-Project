import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_images(images):
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(np.reshape(images[i],(32,32)), cmap='gray')
    
    return plt


def preprocess_images(images):

    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
    kernel_emboss_1 = np.array([[0,-1,-1],
                                [1,0,-1],
                                [1,1,0]])
    kernel_emboss_2 = np.array([[-1,-1,0],
                                [-1,0,1],
                                [0,1,1]])
    kernel_emboss_3 = np.array([[1,0,0],
                                [0,0,0],
                                [0,0,-1]])

    img_shape = images.shape
    norm_images = np.zeros((img_shape[0], img_shape[1], img_shape[2], 1))

    img_min = np.min(images)
    img_max = np.max(images)
    img_mean = np.mean(images)
    delta = img_max - img_min
    for i in range(0, len(images)):
        n_img = images[i]
        n_img = cv2.cvtColor(n_img, cv2.COLOR_RGB2GRAY)
        n_img = (n_img - img_mean)/delta
        n_img = cv2.filter2D(n_img, -1, kernel_sharpen_1)

        norm_images[i] = np.reshape(n_img, (32,32,1))

    return norm_images


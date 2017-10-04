from matplotlib import pyplot as plt
import numpy as np


def plot_images(images, filename=None, cols=4, figsize=(10, 10)):
    rows = np.ceil(len(images) / cols)
    plt.figure(figsize=figsize)

    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.squeeze(image), cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()

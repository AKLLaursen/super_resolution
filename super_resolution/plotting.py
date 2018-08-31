import numpy as np
import matplotlib.pyplot as plt

def display_image_in_actual_size(p, dpi = 200):

    dpi = dpi
    im_data = np.round(p[0]).astype('uint8')
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap = 'gray')

    plt.show()


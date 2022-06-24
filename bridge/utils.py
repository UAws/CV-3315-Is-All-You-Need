import os

import numpy as np
from matplotlib import pyplot as plt


def image_plots(images, captions, rows=1, cols=1, clip=False, normalize=False, gray=True, write_to_file=None, viz=True):
    image_array = None

    index = 0

    fig, axes = plt.subplots(rows, cols, tight_layout=True, figsize=(cols * 6, rows*2))
    # plt.figure(figsize=(20, 15),dpi=100)

    # handle one row and one col case (only on image)
    if rows == 1 and cols == 1:
        axes = [axes]
        images = [images]
        captions = [captions]

    # handle one row or one col case
    if rows == 1 or cols == 1:
        iterator = rows if cols == 1 else cols
        for col in range(iterator):

            if len(images) > index:

                if normalize: images[index] = np.interp(images[index], (images[index].min(), images[index].max()),
                                                        (0, +1))

                # https://stackoverflow.com/a/62839575/14207562
                if clip:
                    axes[col].imshow(np.clip(images[index], 0, 1), cmap='gray' if gray else None)
                else:
                    axes[col].imshow(images[index], cmap='gray' if gray else None)

                axes[col].set_title(captions[index])
                axes[col].set_xlabel(
                    "shape: {}, channels: {}".format(
                        np.shape(images[index]),
                        images[index].shape[2] if len(images[index].shape) == 3 else 1))
            axes[col].axis('off')
            index += 1

        if write_to_file is not None:
            # https://stackoverflow.com/a/34119406/14207562
            results_dir = ''.join([x + '/' for x in write_to_file.split("/")[0:-1]])
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plt.savefig(write_to_file)

        if viz:
            plt.show()
        else:
            # https://matplotlib.org/3.1.1/gallery/misc/agg_buffer_to_array.html
            fig.canvas.draw()
            image_array = np.array(fig.canvas.renderer.buffer_rgba())

        plt.close()

        return image_array

    for row in range(rows):
        for col in range(cols):
            if len(images) > index:

                if normalize: images[index] = np.interp(images[index], (images[index].min(), images[index].max()),
                                                        (0, +1))
                if clip:
                    axes[row, col].imshow(np.clip(images[index], 0, 1), cmap='gray' if gray else None)
                else:
                    axes[row, col].imshow(images[index], cmap='gray' if gray else None)

                axes[row, col].set_title(captions[index])
                axes[row, col].set_xlabel("shape: {}, channels: {}".format(
                    np.shape(images[index]),
                    images[index].shape[2] if len(images[index].shape) == 3 else 1))

            axes[row, col].axis('off')
            index += 1

    if write_to_file is not None:
        results_dir = os.path.abspath(__file__) + ''.join([x + '/' for x in write_to_file.split("/")[0:-1]])
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(write_to_file)

    if viz:
        plt.show()
    else:
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())

    plt.close()

    return image_array
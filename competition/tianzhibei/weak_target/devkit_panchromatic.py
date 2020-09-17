import os

import numpy as np
import rasterio
from rasterio.windows import Window
import cv2

input_data = '/home/data/windowdata/data/tianzhi/tianzhibei/科目1-2/科目1-2/03发布数据-光学-全色.tiff'
out_data = '/home/data/windowdata/data/tianzhi/tianzhibei/科目1-2/科目1-2/test1'
blocksize = 1024
tile_offset = 512


def reshape_as_image(arr):
    """Returns the source array reshaped into the order
    expected by image processing and visualization software
    (matplotlib, scikit-image, etc)
    by swapping the axes order from (bands, rows, columns)
    to (rows, columns, bands)

    Parameters
    ----------
    arr : array-like of shape (bands, rows, columns)
        image to reshape
    """
    # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
    im = np.ma.transpose(arr, [1, 2, 0])
    return im


if __name__ == '__main__':
    with rasterio.open(input_data) as ds:
        width_block = ds.width // blocksize + 1
        height_block = ds.height // blocksize + 1
        for i in range(height_block):
            for j in range(width_block):
                block_xmin = j * tile_offset
                block_ymin = i * tile_offset
                block = np.zeros([3, blocksize, blocksize], dtype=np.uint8)
                img = ds.read(window=Window(block_xmin, block_ymin, blocksize, blocksize))
                block[:, :img.shape[1], :img.shape[2]] = img[:3, :, :]
                block = reshape_as_image(block)
                block = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_data, str(i) + "_" + str(j) + ".png"), block)

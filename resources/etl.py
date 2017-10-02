import os
import glob
import struct
import numpy as np
from PIL import Image


def read_etl_file(f, sample_size, bits_per_pixel, image_shape, unpack_format, image_idx):
    """Reads contents of an ETL file.
    Args:
        f (file): File iterator.
        sample_size (int): Size of sample in bytes.
        bits_per_pixel (int): Bits per pixel.
        image_shape (tuple): Shape of image sample (width-x, height-y).
        unpack_format (str): Format to unpack byte stream (see struct documentation for details).
        image_idx (int): Index of component containing the image.

    Returns:
        Sample data (list of tuples).
    """
    bytestream = f.read(sample_size)
    sample = struct.unpack(unpack_format, bytestream)

    img = Image.frombytes('F', image_shape, sample[image_idx], 'bit', bits_per_pixel)
    img = img.convert('L')
    img = Image.eval(img, lambda x: (255.0 * x) / (2 ** bits_per_pixel - 1))  # rescale to 0~255 range
    sample += (img,)

    return sample


def read_etl6_data(basepath):
    """Reads ETL6 data from specified path.

    Args:
        basepath: Path to ETL6 files.

    Returns:
        ETL6 data (list of tuples).
    """
    paths = glob.glob(os.path.join(basepath, 'ETL6C_*'))

    sample_size = 2052  # bytes
    bits_per_pixel = 4  # bits
    image_shape = (64, 63)  # pixels
    unpack_format = '>H2sH6BI4H4B2H2016s4x'
    image_idx = 20

    data = []
    for path in paths:
        n_samples = os.stat(path).st_size // sample_size

        with open(path, 'r') as f:
            for i in range(n_samples):
                f.seek(i * sample_size)
                data.append(read_etl_file(f, sample_size, bits_per_pixel,
                                          image_shape, unpack_format, image_idx))

    return data


def data2array(data, new_shape=None, norm_factor=255.0, expand_dims=True):
    if new_shape is None:
        images = [np.asarray(d[-1]) for d in data]
    else:
        images = [np.asarray(d[-1].resize(new_shape, Image.BICUBIC)) for d in data]

    array = np.stack(images, axis=0)

    if norm_factor:
        array = np.float32(array) / norm_factor

    if expand_dims:
        array = np.expand_dims(array, axis=3)

    labels = np.char.asarray([d[1] for d in data])

    return array, labels
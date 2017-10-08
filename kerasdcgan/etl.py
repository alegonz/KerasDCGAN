import os
import glob
import struct
import numpy as np
from PIL import Image


def read_etl_sample(f, sample_size, bits_per_pixel, image_shape, unpack_format, charcode_idx, image_idx):
    """Reads a sample from an ETL file.
    Args:
        f (file): File iterator.
        sample_size (int): Size of sample in bytes.
        bits_per_pixel (int): Bits per pixel.
        image_shape (tuple): Shape of image sample (width-x, height-y).
        unpack_format (str): Format to unpack byte stream (see struct documentation for details).
        charcode_idx (int): Index of component containing the character code of the image.
        image_idx (int): Index of component containing the image.

    Returns:
        Data sample (list containing the information of a record).
    """
    bytestream = f.read(sample_size)
    sample = list(struct.unpack(unpack_format, bytestream))

    sample[charcode_idx] = sample[charcode_idx].decode('ascii')

    img = Image.frombytes('F', image_shape, sample[image_idx], 'bit', bits_per_pixel)
    img = img.convert('L')
    img = Image.eval(img, lambda x: (255.0 * x) / (2 ** bits_per_pixel - 1))  # rescale to 0~255 range
    sample += [img]

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
    charcode_idx = 1
    image_idx = 20

    data = []
    for path in paths:
        n_samples = os.stat(path).st_size // sample_size

        with open(path, 'rb') as f:
            for i in range(n_samples):
                f.seek(i * sample_size)
                data.append(read_etl_sample(f, sample_size, bits_per_pixel,
                                            image_shape, unpack_format, charcode_idx, image_idx))

    return data


def data2array(data, new_shape=None, linear_remap=((0, 255), (-1, 1)), expand_dims=True):
    if new_shape is None:
        images = [np.asarray(d[-1]) for d in data]
    else:
        images = [np.asarray(d[-1].resize(new_shape, Image.BICUBIC)) for d in data]

    array = np.stack(images, axis=0)

    if linear_remap:

        (x0, x1), (y0, y1) = linear_remap
        if x0 == x1 or y0 == y1:
            raise ValueError('Invalid remap values.')

        m = (y1 - y0) / (x1 - x0)
        array = m * (np.float32(array) - x0) + y0

    if expand_dims:
        array = np.expand_dims(array, axis=3)

    labels = np.char.asarray([d[1] for d in data])

    return array, labels

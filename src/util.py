import numpy as np
import gzip
import struct


def unzip(file_path: str) -> bytes:
    with gzip.open(file_path, "rb") as f_in:
        # Read the uncompressed data
        uncompressed_data = f_in.read()
        return uncompressed_data


def parse_images(file_path: str) -> np.ndarray[np.float32]:
    data: bytes = unzip(file_path)
    image_num: int = int.from_bytes(data[4:8], byteorder='big')
    row_num: int = int.from_bytes(data[8:12], byteorder='big')
    col_num: int = int.from_bytes(data[12:16], byteorder='big')
    data = data[16:]
    assert(row_num == 28 and col_num == 28 and len(data) % image_num == 0)
    image_size = int(len(data)/image_num)
    images = np.ndarray((image_num, image_size), dtype=np.float32)
    for i in range(image_num):
        tup = struct.unpack_from("B"*image_size, data, i*image_size)
        images[i] = np.array(tup).astype(np.float32)
    images_normed = (images - np.min(images)) / (np.max(images) - np.min(images))
    return images_normed


def parse_labels(file_path: str) -> np.ndarray[np.uint8]:
    data: bytes = unzip(file_path)
    label_num: int = int.from_bytes(data[4:8], byteorder='big')
    data = data[8:]
    labels = np.ndarray((label_num, ), dtype=np.uint8)
    for i in range(label_num):
        labels[i] = struct.unpack_from("B", data, i)[0]
    return labels

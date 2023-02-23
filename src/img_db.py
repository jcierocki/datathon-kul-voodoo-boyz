import numpy as np
import cv2
import os
import math
import sys

from copy import deepcopy
from multiprocessing import Pool, cpu_count
from functools import partial

from typing import Literal
from numpy.typing import NDArray

from scipy.fft import dct, fft

def crop(m: cv2.Mat, crop_size: int):
    img_width, img_height = m.shape
    img_x0 = math.floor((img_width - crop_size) / 2)
    img_y0 = math.floor((img_height - crop_size) / 2)

    return m[img_x0:(img_x0 + crop_size), img_y0:(img_y0 + crop_size)]

class ImageDatabase(object):
    def __init__(
            self, 
            path_original: str, 
            path_generated: str, 
            n_threads: int = cpu_count() - 2
        ):
        self.path_original = path_original
        self.path_generated = path_generated
        self.n_threads = n_threads
        self.images: None | list[cv2.Mat | NDArray[np.double]] = None
        self.originality_flags: None | NDArray[np.bool_] = None

    __crop = staticmethod(crop)

    def load_files(self, max_obs: int = sys.maxsize):
        loader = partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
        files_original = [ os.path.join(self.path_original, file) for file in os.listdir(self.path_original) if file.endswith(".jpg") or file.endswith(".png") ]
        files_generated = [ os.path.join(self.path_generated, file) for file in os.listdir(self.path_generated) if file.endswith(".jpg") or file.endswith(".png") ]

        with Pool(self.n_threads) as p:
            imgs_original = p.map(loader, files_original[:max_obs])
            imgs_generated = p.map(loader, files_generated[:max_obs])

        self.images = imgs_original + imgs_generated
        self.originality_flags = np.concatenate([
            np.ones(len(imgs_original), dtype=np.bool_),
            np.zeros(len(imgs_generated), dtype=np.bool_)
        ]) 

    def get_data(self, copy: bool = False) -> tuple[list[cv2.Mat], NDArray[np.bool_]]:
        if self.images is None or self.originality_flags is None:
            raise Exception("You need to load the data first before transforming.")
        
        if copy:
            return self.images, self.originality_flags
        else:
            return deepcopy(self.images), np.copy(self.originality_flags)
        
    def denoise(self) -> None:
        if self.images is None or self.originality_flags is None:
            raise Exception("You need to load the data first before transforming.")
        
        denoiser = partial(cv2.fastNlMeansDenoising, dst=None, h=7, templateWindowSize=11, searchWindowSize=40)

        with Pool(self.n_threads) as p:
            self.images = p.map(denoiser, self.images)

    def crop(self, crop_size : int = 64) -> None:
        if self.images is None or self.originality_flags is None:
            raise Exception("You need to load the data first before transforming.")

        with Pool(self.n_threads) as p:
            self.images = p.map(partial(self.__crop, crop_size=crop_size), self.images)
    
    def export(
            self, 
            tranformation: Literal["dct", "fft"] = "dct",
            log_abs: bool = True, 
        ) -> tuple[NDArray[np.double], NDArray[np.bool_]]:

        if self.images is None or self.originality_flags is None:
            raise Exception("You need to load the data first before transforming.")
        
        feature_mat = np.matrix([m.flatten('C') for m in self.images])

        if tranformation == "dct":
            feature_mat = dct(feature_mat, workers=self.n_threads)
        else:
            feature_mat = fft(feature_mat, workers=self.n_threads)

        if log_abs:
            feature_mat = np.log(np.abs(feature_mat + 1))

        return feature_mat, self.originality_flags


if __name__ == "__main__":
    img_db = ImageDatabase("data/images/original", "data/images/generated")
    img_db.load_files()

    img_db.denoise()
    img_db.crop()

    X, y = img_db.export(log_abs=True)

    print(X)
    print(y)

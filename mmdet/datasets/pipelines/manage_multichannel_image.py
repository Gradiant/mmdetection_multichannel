# Copyright (c) Gradiant. All rights reserved.
import os.path as osp

import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from cv2 import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_IGNORE_ORIENTATION,
    IMREAD_UNCHANGED,
)
from mmdet.core import BitmapMasks, PolygonMasks
from skimage import io, transform

from ..builder import PIPELINES
from . import transforms

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadMultiChannelImgFromFile:
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        file_client_args=dict(backend="disk"),
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results["img_prefix"] is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        img = io.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        img = np.moveaxis(img, 0, -1)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]

        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"file_client_args={self.file_client_args})"
        )
        return repr_str


@PIPELINES.register_module()
class ResizeMultiChannel(transforms.Resize):
    def _resize_img(self, results):

        img_shape = results["img"].shape
        img = transform.resize(results["img"], self.img_scale[0])

        w_scale = img.shape[1] / img_shape[1]
        h_scale = img.shape[2] / img_shape[2]

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape  # in case that there is no padding
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio


@PIPELINES.register_module()
class NormalizeMultiChannel(transforms.Normalize):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (float): Mean value.
        std (float): Std value.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        mean_array = np.array(0, dtype=np.float32)
        std_array = np.array(0, dtype=np.float32)

        for key in results.get("img_fields", ["img"]):
            mean_array = self.mean * np.ones(results[key].shape, dtype=np.float32)
            std_array = self.std * np.ones(results[key].shape, dtype=np.float32)
            # results[key] = mmcv.imnormalize(results[key], mean_array, std_array,False)
            img = results[key]
            img = np.subtract(img, mean_array)
            img = np.divide(img, std_array)
            results[key] = img

        results["img_norm_cfg"] = dict(mean=mean_array, std=std_array, to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=Always false)"
        return repr_str
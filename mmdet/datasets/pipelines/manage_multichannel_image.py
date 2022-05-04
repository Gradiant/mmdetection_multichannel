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
from . import transforms, auto_augment

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

_MAX_LEVEL = 10

def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b

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
class BrightnessTransformMultiChannel(auto_augment.BrightnessTransform):
    
    def __init__(self, level, prob=0.5, dims=[]):

        assert isinstance(level, (int, float, list)), \
            'The level must be type list, int or float.'
        assert isinstance(dims, (list)), \
            'dims must be list of channels'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'

        if isinstance(level, (list)):
            if isinstance(dims, list) and len(dims) != 0:
                assert len(level)==len(dims), \
                    'Level list length should match dimension list length'
            for l in level:
                assert 0 <= l <= _MAX_LEVEL, \
                    'The level should be in range [0,_MAX_LEVEL].'
        else:
            assert 0 <= level <= _MAX_LEVEL, \
                'The level should be in range [0,_MAX_LEVEL].'
        
        self.level = level
        self.prob = prob
        self.dims = dims

    def __call__(self, results):
        """Call function for Brightness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results

        original_img = results['img']
        
        assert len(self.dims) <= original_img.shape[-1], \
            'Selected channels can\'t be greater than numer of channels' 
        
        for d in self.dims:
            assert d <= (original_img.shape[-1]-1) , \
            f'Channel must be one of {range(0, original_img.shape[-1]-1)} but found {d}' 

        if len(self.dims) != 0:

            if isinstance(self.level, list):

                for l, d in zip(self.level, self.dims):
                    results['img'] = original_img[:,:,d]
                    self._adjust_brightness_img(results, enhance_level_to_value(l))
                    original_img[:,:,d] = results['img']
            
            else:
                results['img'] = original_img[:,:,self.dims]
                self._adjust_brightness_img(results, enhance_level_to_value(self.level))
                original_img[:,:,self.dims] = results['img']
        
        else:
            if isinstance(self.level, list):
                assert len(self.level) == original_img.shape[-1], \
                    'When type(level)==list, len(level) should match total number of channels' 

                for d, l in enumerate(self.level):
                    results['img'] = original_img[:,:,d]
                    self._adjust_brightness_img(results, enhance_level_to_value(l))
                    original_img[:,:,d] = results['img']
            else:
                self._adjust_brightness_img(results, enhance_level_to_value(self.level))

        results['img'] = original_img

        return results

@PIPELINES.register_module()
class NormalizeMinMaxChannelwise:
    """Normalize the image channelwise.
    """

    def __init__(self):
        print("Normalizing array between 0 and 1 per channel")

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):

            for c in range(0, results[key].shape[-1]):
                channel = results[key][:,:,c]
                channel -= np.min(channel)
                channel /= np.max(channel)

                results[key][:,:,c]=channel
                
        return results


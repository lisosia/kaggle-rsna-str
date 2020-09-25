import numpy as np
from scipy.ndimage.interpolation import zoom


def trans_ind_3d(img):
    """
    assume input (D,256,256)
    random crop h,w=(224,224)
    interpolate to (128,224,224)
    """
    HW = 224
    D, H, W = img.shape
    assert H == 256 and W == 256
    hs, ws = np.random.randint(0, 256 - HW, size=2)
    img = img[:, hs:hs+HW, ws:ws+HW]
    # order=1 linear. order=0 nearest
    # https://stackoverflow.com/questions/57777370/set-interpolation-method-in-scipy-ndimage-map-coordinates-to-nearest-and-bilinea
    img = zoom(img, (128. / D, 1, 1), order=1)
    return img

def trans_ind_3d_valid(img):
    """
    assume input (D,256,256)
    random crop h,w=(224,224)
    interpolate to (128,224,224)
    """
    HW = 224
    D, H, W = img.shape
    assert H == 256 and W == 256
    hs = ws = (256 - HW) // 2
    img = img[:, hs:hs+HW, ws:ws+HW]
    img = zoom(img, (128. / D, 1, 1), order=1)
    return img

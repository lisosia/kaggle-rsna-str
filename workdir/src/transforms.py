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


def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(np.clip(W * cut_rat, 0, W - 1))
    cut_h = np.int(np.clip(H * cut_rat, 0, H - 1))

    # uniform
    cx = np.random.randint(W - cut_w)
    cy = np.random.randint(H - cut_h)

    bbx1 = np.clip(cx         ,0, W)
    bby1 = np.clip(cy         ,0, H)
    bbx2 = np.clip(cx + cut_w, 0, W)
    bby2 = np.clip(cy + cut_h, 0, H)

    real_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))   # NOT-cutted area / total area == positive portion
    return bbx1, bby1, bbx2, bby2, real_lam

# assume binary label
# img: C,H,W
# overwrite img1
def cutmix(img1, img2, y1: dict, y2: dict, alpha=1.5):
    _, H, W = img1.shape[:3]
    lam_pre = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2, real_lam = rand_bbox(H, W, lam_pre)
    img1[:, bby1 : bby2, bbx1: bbx2] = img2[:, bby1 : bby2, bbx1: bbx2]

    # new_y = real_lam * y1 + (1 - real_lam) * y2
    new_y = dict([
        (key, real_lam * y1[key] + (1 - real_lam) * y2[key]
        ) for key in y1.keys()])
    return img1, new_y


if __name__=="__main__":
    # test cutmix
    import numpy as np
    import matplotlib.pyplot as plt
    shape = (3, 224,244)
    img1 = np.ones(shape)
    img2 = np.zeros(shape)
    img, y = cutmix(img1, img2, {"y0": 0}, {"y0": 1})
    plt.imshow(img.transpose(1,2,0))
    plt.title(f"new_y = {y['y0']:.3f} (black==1)")
    plt.show()

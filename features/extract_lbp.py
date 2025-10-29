from skimage.feature import local_binary_pattern
import numpy as np

def extract_lbp_features(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist
from skimage.feature import hog

def extract_hog_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

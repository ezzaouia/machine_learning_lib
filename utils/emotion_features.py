import os
import shutil
from distutils import dir_util
import glob
import skimage.io
import skimage.color
import skimage.io
import skimage.transform
import numpy as np
import cv2
import dlib


class EmotionFeatures(object):
    
    def __init__(self, Xtr):
        self.X = Xtr
        module_path = os.path.split(os.path.dirname(os.path.realpath('__file__')))[0]
        predictor_path = os.path.join(module_path, 'machine_learning_lib' , 'resources',
                                       'shape_predictor_68_face_landmarks.dat') 
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        if self.predictor == None:
            print classifier_path
            raise Exception('Predictor dat file was not found.')
    
    def face_landmarks_features(self):
        # Face landmark dimension reduction
        # compute face landmarks for each sample
        # in the dataset and return landmark featured dataset
        return np.array([self.get_landmarks_for_single_image(x).flatten() for x in self.X], 'f')
    
    def get_landmarks_for_single_image(self, image):
        rects = self.detector(image, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return np.array([[p.x, p.y] for p in self.predictor(image, rects[0]).parts()])

    def pca_features(self):
        # Face PCA dim. reduction
        # compute PCA for each sample in the dataset
        # and return PCA featured dataset
        n_components = 150
        pca = RandomizedPCA(n_components=n_components, whiten=True).fit(self.X)
        eigenfaces = pca.components_.reshape((n_components, 100, 100))
        X_train_pca = pca.transform(self.X)
        return X_train_pca
    
    def pca_features_new(self):
        # Assume input data matrix X of size [N x D]  here D = 100 * 100
        X = self.X
        X -= np.mean(self.X, axis = 0) # zero-center the data (important)
        cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
        U,S,V = np.linalg.svd(cov)
        Xrot = np.dot(X, U) # decorrelate the data
        Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
        return Xrot_reduced
    
    def lbp_features(self):
        return np.array([self.lbp_describe(x) for x in self.X], 'f')
    
    def lbp_describe(self, image, eps=1e-7, numPoints=24, radius=8):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, numPoints,
            radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins = np.arange(0, numPoints + 2),
            range=(0, numPoints + 1))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
# coding: utf-8
# Author：Qi Tang
# Date ：02/10/2022

import cv2
import numpy as np
import math
import utils
import os.path


# A simple convolution function that returns the filtered images.
def getFilterImages(filters, img):
    featureImages = []
    for filter in filters:
        kern, params = filter
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        featureImages.append(fimg)
    return featureImages


# Apply the R^2 threshold technique here, note we find energy in the spatial domain.
def filterSelection(featureImages, threshold, img, howManyFilterImages):
    idEnergyList = []
    id = 0
    height, width = img.shape
    for featureImage in featureImages:
        thisEnergy = 0.0
        for x in range(height):
            for y in range(width):
                thisEnergy += pow(np.abs(featureImage[x][y]), 2)
        idEnergyList.append((thisEnergy, id))
        id += 1
    E = 0.0
    for E_i in idEnergyList:
        E += E_i[0]
    sortedlist = sorted(idEnergyList, key=lambda energy: energy[0], reverse=True)

    tempSum = 0.0
    RSquared = 0.0
    added = 0
    outputFeatureImages = []
    while ((RSquared < threshold) and (added < howManyFilterImages)):
        tempSum += sortedlist[added][0]
        RSquared = (tempSum / E)
        outputFeatureImages.append(featureImages[sortedlist[added][1]])
        added += 1
    return outputFeatureImages


# This is where we create the gabor kernel
# Feel free to uncomment the other list of theta values for testing.
def build_filters(lambdas, ksize, gammaSigmaPsi):
    filters = []
    thetas = []

    # Thetas 1
    # -------------------------------------
    thetas.extend([0, 45, 90, 135])

    # Thetas2
    # -------------------------------------
    # thetas.extend([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

    thetasInRadians = [np.deg2rad(x) for x in thetas]

    for lamb in lambdas:
        for theta in thetasInRadians:
            params = {'ksize': (ksize, ksize), 'sigma': gammaSigmaPsi[1], 'theta': theta, 'lambd': lamb,
                      'gamma': gammaSigmaPsi[0], 'psi': gammaSigmaPsi[2], 'ktype': cv2.CV_64F}
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append((kern, params))
    return filters


# Here is where we convert radial frequencies to wavelengths.
# Feel free to uncomment the other list of lambda values for testing.
def getLambdaValues(img):
    height, width = img.shape

    # calculate radial frequencies.
    max = (width / 4) * math.sqrt(2)
    min = 4 * math.sqrt(2)
    temp = min
    radialFrequencies = []

    # Lambda 1
    # -------------------------------------
    while (temp < max):
        radialFrequencies.append(temp)
        temp = temp * 2

    # Lambda 2
    # -------------------------------------
    # while(temp < max):
    #     radialFrequencies.append(temp)
    #     temp = temp * 1.5

    radialFrequencies.append(max)
    lambdaVals = []
    for freq in radialFrequencies:
        lambdaVals.append(width / freq)
    return lambdaVals


# The activation function with gaussian smoothing
def nonLinearTransducer(img, gaborImages, L, sigmaWeight, filters):
    alpha_ = 0.25
    featureImages = []
    count = 0
    for gaborImage in gaborImages:

        # Spatial method of removing the DC component
        avgPerRow = np.average(gaborImage, axis=0)
        avg = np.average(avgPerRow, axis=0)
        gaborImage = gaborImage.astype(float) - avg

        # gaborImage = cv2.normalize(gaborImage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Normalization sets the input to the active range [-2,2] this becomes [-8,8] with alpha_
        gaborImage = cv2.normalize(gaborImage, gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)

        height, width = gaborImage.shape
        copy = np.zeros(img.shape)
        for row in range(height):
            for col in range(width):
                # centralPixelTangentCalculation_bruteForce(gaborImage, copy, row, col, alpha, L)
                copy[row][col] = math.fabs(math.tanh(alpha_ * (gaborImage[row][col])))

        # now apply smoothing
        copy, destroyImage = applyGaussian(copy, L, sigmaWeight, filters[count])
        if (not destroyImage):
            featureImages.append(copy)
        count += 1

    return featureImages


# I implemented this just for completeness
# It just applies the tanh function and smoothing as spatial convolution
def centralPixelTangentCalculation_bruteForce(img, copy, row, col, alpha, L):
    height, width = img.shape
    windowHeight, windowWidth, inita, initb = \
        utils.getRanges_for_window_with_adjust(row, col, height, width, L)

    sum = 0.0
    for a in range(windowHeight + 1):
        for b in range(windowWidth + 1):
            truea = inita + a
            trueb = initb + b
            sum += math.fabs(math.tanh(alpha * (img[truea][trueb])))

    copy[row][col] = sum / pow(L, 2)


# Apply Gaussian with the central frequency specification
def applyGaussian(gaborImage, L, sigmaWeight, filter):
    height, N_c = gaborImage.shape

    nparr = np.array(filter[0])
    u_0 = nparr.mean(axis=0)
    u_0 = u_0.mean(axis=0)

    destroyImage = False
    sig = 1
    if (u_0 < 0.000001):
        print("div by zero occured for calculation:")
        print("sigma = sigma_weight * (N_c/u_0), sigma will be set to zero")
        print("removing potential feature image!")
        destroyImage = True
    else:
        sig = sigmaWeight * (N_c / u_0)

    return cv2.GaussianBlur(gaborImage, (L, L), sig), destroyImage


# Remove feature images with variance lower than 0.0001
def removeFeatureImagesWithSmallVariance(featureImages, threshold):
    toReturn = []
    for image in featureImages:
        if (np.var(image) > threshold):
            toReturn.append(image)

    return toReturn


# Our main driver function to return the segmentation of the input image.
def runGabor(image_path, save_path, k):
    if (not os.path.isfile(image_path)):
        print(image_path, " is not a file!")
        exit(0)


    M_transducerWindowSize = 31
    if ((M_transducerWindowSize % 2) == 0):
        print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1

    k_clusters = k
    k_gaborSize = 19

    spatialWeight = 2
    gammaSigmaPsi = []
    gammaSigmaPsi.append(0.5)  # gamma
    gammaSigmaPsi.append(7)  # sigma
    gammaSigmaPsi.append(0)  # psi
    variance_Threshold = 0.0001  # vt
    howManyFeatureImages = 100  # fi
    R_threshold = 0.95  # R
    sigmaWeight = 0.5  # siw

    img = cv2.imread(image_path, 0)

    lambdas = getLambdaValues(img)
    filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)

    print("Gabor kernels created, getting filtered images")
    filteredImages = getFilterImages(filters, img)
    filteredImages = filterSelection(filteredImages, R_threshold, img, howManyFeatureImages)

    print("Applying nonlinear transduction with Gaussian smoothing")
    featureImages = nonLinearTransducer(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
    featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)

    featureVectors = utils.constructFeatureVectors(featureImages, img)
    featureVectors = utils.normalizeData(featureVectors, False, spatialWeight=spatialWeight)

    print("Clustering...")
    labels = utils.clusterFeatureVectors(featureVectors, k_clusters)
    utils.printClassifiedImage(labels, k_clusters, img, save_path)

import math
import numpy as np
import sklearn.cluster as clstr
import cv2
import os
import matplotlib.pyplot as pyplt
import scipy.cluster.vq as vq
import argparse


# This is the function that checks boundaries when performing spatial convolution.
def getRanges_for_window_with_adjust(row, col, height, width, W):
    mRange = []
    nRange = []

    mRange.append(0)
    mRange.append(W - 1)

    nRange.append(0)
    nRange.append(W - 1)

    initm = int(round(row - math.floor(W / 2)))
    initn = int(round(col - math.floor(W / 2)))

    if (initm < 0):
        mRange[1] += initm
        initm = 0

    if (initn < 0):
        nRange[1] += initn
        initn = 0

    if (initm + mRange[1] > (height - 1)):
        diff = ((initm + mRange[1]) - (height - 1))
        mRange[1] -= diff

    if (initn + nRange[1] > (width - 1)):
        diff = ((initn + nRange[1]) - (width - 1))
        nRange[1] -= diff

    windowHeight = mRange[1] - mRange[0]
    windowWidth = nRange[1] - nRange[0]

    return int(round(windowHeight)), int(round(windowWidth)), int(round(initm)), int(round(initn))


# Used to normalize data before clustering occurs.
# Whiten sets the variance to be 1 (unit variance),
# spatial weighting also takes place here.
# The mean can be subtracted if specified by the implementation.
def normalizeData(featureVectors, setMeanToZero, spatialWeight=1):
    means = []
    for col in range(0, len(featureVectors[0])):
        colMean = 0
        for row in range(0, len(featureVectors)):
            colMean += featureVectors[row][col]
        colMean /= len(featureVectors)
        means.append(colMean)

    for col in range(2, len(featureVectors[0])):
        for row in range(0, len(featureVectors)):
            featureVectors[row][col] -= means[col]
    copy = vq.whiten(featureVectors)
    if (setMeanToZero):
        for row in range(0, len(featureVectors)):
            for col in range(0, len(featureVectors[0])):
                copy[row][col] -= means[col]

    for row in range(0, len(featureVectors)):
        copy[row][0] *= spatialWeight
        copy[row][1] *= spatialWeight

    return copy


# Create the feature vectors and add in row and column data
def constructFeatureVectors(featureImages, img):
    featureVectors = []
    height, width = img.shape
    for row in range(height):
        for col in range(width):
            featureVector = []
            featureVector.append(row)
            featureVector.append(col)
            for featureImage in featureImages:
                featureVector.append(featureImage[row][col])
            featureVectors.append(featureVector)

    return featureVectors


# An extra function if we are looking to save our feature vectors for later
def printFeatureVectors(outDir, featureVectors):
    f = open(outDir, 'w')
    for vector in featureVectors:
        for item in vector:
            f.write(str(item) + " ")
        f.write("\n")
    f.close()


# If we want to read in some feature vectors instead of creating them.
def readInFeatureVectorsFromFile(dir):
    list = [line.rstrip('\n') for line in open(dir)]
    list = [i.split() for i in list]
    newList = []
    for row in list:
        newRow = []
        for item in row:
            floatitem = float(item)
            newRow.append(floatitem)
        newList.append(newRow)

    return newList


# Print the final result, the user can also choose to make the output grey
def printClassifiedImage(labels, k, img, outdir):
    labels = labels.reshape(img.shape)
    max = np.max(labels)
    for row in range(0, len(labels)):
        for col in range(0, len(labels[0])):
            outputIntensity = 255 * labels[row][col] / max
            labels[row][col] = outputIntensity
    if not os.path.exists("./segmentation_results/"):
        os.mkdir("./segmentation_results/")
    cv2.imwrite(outdir, labels.reshape(img.shape))


# Call the k means algorithm for classification
def clusterFeatureVectors(featureVectors, k):
    kmeans = clstr.KMeans(n_clusters=k)
    kmeans.fit(featureVectors)
    labels = kmeans.labels_

    return labels


# To clean up old filter and feature images if the user chose to print them.
def deleteExistingSubResults(outputPath):
    for filename in os.listdir(outputPath):
        if (filename.startswith("filter") or filename.startswith("feature")):
            os.remove(filename)


# Checks user input (i.e. cannot have a negative mask size value)
def check_positive_int(n):
    int_n = int(n)
    if int_n < 0:
        raise argparse.ArgumentTypeError("%s is negative" % n)
    return int_n


# Checks user input (i.e. cannot have a negative weighting value)
def check_positive_float(n):
    float_n = float(n)
    if float_n < 0:
        raise argparse.ArgumentTypeError("%s is negative " % n)
    return float_n

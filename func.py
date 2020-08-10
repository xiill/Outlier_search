import statistics
import numpy as np
import scipy as sp
import math
from scipy.spatial.distance import cdist

"""Мат ожидание по всем кадрам"""
"""i - одна из 16 точек """
def math_expect(src_point, i):
    count_of_frames = len(src_point)
    x = [0] * count_of_frames
    y = [0] * count_of_frames

    for j in range(0, count_of_frames):
        x[j] = src_point[j][0][i][0]
        y[j] = src_point[j][0][i][1]

    return [np.mean(x), np.mean(y)]

"""Дисперсия"""
def variance(src_point, i):
    count_of_frames = len(src_point)
    x = [0] * count_of_frames
    y = [0] * count_of_frames

    for j in range(0, count_of_frames):
        x[j] = src_point[j][0][i][0]
        y[j] = src_point[j][0][i][1]

    return [np.var(x), np.var(y)]

"""Ковариационная матрица"""
def covariance (src_point, i):
    count_of_frames = len(src_point)
    x = [0] * count_of_frames
    y = [0] * count_of_frames

    for j in range(0, count_of_frames):
        x[j] = src_point[j][0][i][0]
        y[j] = src_point[j][0][i][1]

    return np.cov(x, y)

"""Расстояние Махалонобиса"""
def mahalanobisa(xy, i, src_point):
    inv_covmat = sp.linalg.inv(covariance(src_point, i))
    math_expected = math_expect(src_point, i)
    m = np.array([[math_expected[0]], [math_expected[1]]])
    mat = abs(xy - m)
    mah = (mat.T).dot(inv_covmat)
    return math.sqrt(mah.dot(mat))

def outlier(p, i, src_point):
    count = 0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    count_of_frames = len(src_point)
    with open(str(i) + '_point_mahalanobis.txt', 'a+') as file:
        for j in range(0, count_of_frames):
            xy = np.array([[src_point[j][0][i][0]], [src_point[j][0][i][1]]])
            mahal = mahalanobisa(xy, i, src_point)
            if (mahal > t):
                count = count + 1
                file.write(str(j) + " \n")
    with open('count_mahal.txt', 'a+') as f:
        f.write(str(count) + " \n")

def three_sigma(xy, i, src_point):
    sigma = [0] * 2
    math_expected = math_expect(src_point, i)
    m = np.array([[math_expected[0]], [math_expected[1]]])
    k = abs(xy - m)
    var = variance(src_point, i)
    sigma[0] = k[0] / math.sqrt(var[0])
    sigma[1] = k[1] / math.sqrt(var[1])
    return sigma

def sigma_rule( i, src_point):
    count = 0
    count_of_frames = len(src_point)
    with open(str(i) + '_point_sigma.txt', 'a+') as file:
        for j in range(0, count_of_frames):
            xy = np.array([[src_point[j][0][i][0]], [src_point[j][0][i][1]]])
            sigma = three_sigma(xy, i, src_point)
            if (sigma[0] > 3 or sigma[1] > 3):
                count = count + 1
                file.write(str(j) + " \n")
    with open('count_sigma.txt', 'a+') as f:
        f.write(str(count) + " \n")
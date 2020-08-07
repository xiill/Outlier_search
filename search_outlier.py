import json
from func import *

src_point = []

""" Считываем кортежи точек src """
for line in open('C:/Users/tansa/Desktop/src_points.json', 'r'):
    src_point.append(json.loads(line))

count_of_frames = len(src_point)

"""Нормирование"""
for j in range(0, count_of_frames):
    mean_x = 0
    mean_y = 0
    for i in range(16):
        mean_x = mean_x + src_point[j][0][i][0]
        mean_y = mean_y + src_point[j][0][i][1]
    """ Берем один кортеж и из него генерируем x и y """
    for i in range(16):
        src_point[j][0][i][0] = src_point[j][0][i][0] - mean_x / 16
        src_point[j][0][i][1] = src_point[j][0][i][1] - mean_y / 16

math_expected = [0] * 16
var = [0] * 16
cov = [0] * 16
out = [0] * 16
sigma = [0] * 16

p = 0.9973

for i in range(16):
    """Мат ожидание"""
    math_expected[i] = math_expect(src_point, i)
    """Дисперсия"""
    var[i] = variance(src_point, i)
    """Ковариационная матрица"""
    cov[i] = covariance(src_point, i)
    """Расстояние Махаланобиса - Поиск выбросов"""
    out[i] = outlier(p, i, src_point)
    """Правило трех сигм"""
    sigma[i] = sigma_rule(i, src_point)


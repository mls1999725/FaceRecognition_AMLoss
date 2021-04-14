
import os
import pickle
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np
import math

# xsm1 = (180.0/math.pi) * pickle.load(open("/home/baixy/Codes/class-invariant-loss/xarc1.pkl", 'rb'))
# xsm10 = (180.0/math.pi) * pickle.load(open("/home/baixy/Codes/class-invariant-loss/xarc10.pkl", 'rb'))
# xsm20 = (180.0/math.pi) * pickle.load(open("/home/baixy/Codes/class-invariant-loss/xarc20.pkl", 'rb'))


# fig, ax = plt.subplots(nrows=3, ncols=2)
# plt.xlim(0,180)
# # ax.set_ylim([0,180])
# # ax0.hist(x1, 100, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# ax[0][0].hist(xsm1, 100, range=(0,180), normed=1, histtype='bar', facecolor='blue', alpha=0.5)
# ax[1][0].hist(xsm10, 100, range=(0,180), normed=1, histtype='bar', facecolor='pink', alpha=0.75)
# ax[2][0].hist(xsm20, 100, range=(0,180), normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# plt.savefig("./Figures/xarc.png")

# x1 = (180.0/math.pi) * pickle.load(open("/home/baixy/Codes/class-invariant-loss/yarc1.pkl", 'rb'))
# x10 = (180.0/math.pi) * pickle.load(open("/home/baixy/Codes/class-invariant-loss/yarc10.pkl", 'rb'))
# x20 = (180.0/math.pi) * pickle.load(open("/home/baixy/Codes/class-invariant-loss/yarc20.pkl", 'rb'))

# # ax0.hist(x1, 100, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# ax[0][1].hist(x1, 100, range=(0,180), normed=1, histtype='bar', facecolor='blue', alpha=0.5)
# ax[1][1].hist(x10, 100,range=(0,180),  normed=1, histtype='bar', facecolor='pink', alpha=0.75)
# ax[2][1].hist(x20, 100,range=(0,180),  normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# plt.show()

# Target angle
# ax1 = plt.subplot(3, 1, 1)
# ax1.hist(xsm1, 100, range=(0,180), normed=1, histtype='bar', facecolor='blue', alpha=0.5)
# plt.ylabel("Percentage")
# ax2 = plt.subplot(3, 1, 2)
# ax2.hist(xsm10, 100, range=(0,180), normed=1, histtype='bar', facecolor='pink', alpha=0.75)
# plt.ylabel("Percentage")
# ax3 = plt.subplot(3, 1, 3)
# ax3.hist(xsm20, 100, range=(0,180), normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# plt.xlabel("Theta")
# plt.ylabel("Percentage")
# plt.savefig("./Figures/xarc.png")

# Untarget angle
# ax1 = plt.subplot(3, 1, 1)
# ax1.hist(x1, 100, range=(0,180), normed=1, histtype='bar', facecolor='blue', alpha=0.5)
# # plt.xlabel("Thera")
# plt.ylabel("Percentage")
# ax2 = plt.subplot(3, 1, 2)
# ax2.hist(x10, 100,range=(0,180),  normed=1, histtype='bar', facecolor='pink', alpha=0.75)
# # plt.xlabel("Thera")
# plt.ylabel("Percentage")
# ax3 = plt.subplot(3, 1, 3)
# ax3.hist(x20, 100,range=(0,180),  normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# plt.xlabel("Theta")
# plt.ylabel("Percentage")
# plt.savefig("./Figures/yarc.png")

# plot loss curve
# x = np.arange(0, 180, 0.001)
# x_arc = x / 180 * np.pi
# y_h = 0.5*(1 - np.cos(x_arc) * np.cos(x_arc))
# y_g = 0.5*np.cos(x_arc)*np.cos(x_arc)
# plt.plot(x, y_h)
# plt.plot(x, y_g)
# plt.legend(['h', 'g'], loc='upper right')
# plt.xlabel("Theta")
# plt.ylabel("Adaptive margin")
# plt.savefig("./Figures/loss.png")

# plot target logit curve
# softmax = np.cos(x_arc)
# cosface = np.cos(x_arc) - 0.25
# am = np.cos(x_arc) - y_h
# plt.plot(x, softmax)
# plt.plot(x, cosface)
# plt.plot(x, am)
# plt.legend(['Softmax', 'Cosface', 'AM-Loss'])
# plt.savefig("./Figures/Target_logit.png")

# plot Untarget logit curve
# softmax = np.cos(x_arc)
# cosface = np.cos(x_arc) + 0.25
# am = np.cos(x_arc) + y_g
# plt.plot(x, softmax)
# plt.plot(x, cosface)
# plt.plot(x, am)
# plt.legend(['Softmax', 'Cosface', 'AM-Loss'])
# plt.savefig("./Figures/UnTarget_logit.png")

# plot softmax
# x = np.arange(-6, 6, 0.01)
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x), axis=0)
# y = softmax(x)
# plt.plot(x, y)
# plt.savefig("./Figures/softmax.png")

# plot loss curve
loss = []
epochloss = [[21.553895, 20.207734, 19.673554, 19.277869, 18.466493, 16.824784, 14.286705, 11.781200],
[9.720930, 9.427163, 9.224884, 9.078868, 8.951858, 8.907770, 8.834727, 8.829289],
[8.710565, 8.703917, 8.700847, 8.659426, 8.633184, 8.594570, 8.562628, 8.541521],
[8.394015, 8.386749, 8.330597, 8.292768, 8.262853, 8.193026, 8.123724, 8.076604],
[7.942103, 8.009807, 7.912449, 7.870341, 7.850980, 7.794432, 7.749234, 7.724153],
[7.486394, 7.542279, 7.518716, 7.480445, 7.409077, 7.402428, 7.310614, 7.286966],
[7.079324, 7.102076, 7.079887, 7.015341, 6.998482, 6.934735, 6.906130, 6.879927], 
[6.606092, 6.685705, 6.673167, 6.615455, 6.620867, 6.598608, 6.543891, 6.572778], 
[6.301093, 6.341373, 6.370884, 6.342672, 6.285531, 6.295593, 6.233673, 6.235902], 
[5.920603, 6.063984, 6.038892, 5.995556, 5.981678, 5.996657, 5.974102, 5.941454],
[5.669005, 5.737569, 5.803087, 5.824238, 5.764738, 5.729704, 5.701132, 5.714019], 
[5.363488, 5.477239, 5.533708, 5.570642, 5.551744, 5.463392, 5.489148, 5.529861], 
[5.136290, 5.360043, 5.335773, 5.369426, 5.324254, 5.306660, 5.286714, 5.281353], 
[4.937661, 5.105238, 5.185729, 5.163730, 5.148630, 5.177537, 5.120115, 5.133715], 
[4.793229, 4.938173, 5.020413, 5.058847, 4.957197, 4.991799, 4.983587, 4.959401], 
[4.622469, 4.800453, 4.876630, 4.867699, 4.831925, 4.868887, 4.816074, 4.806034], 
[4.464675, 4.641731, 4.715689, 4.780161, 4.787895, 4.740182, 4.775801, 4.776006], 
[3.679364, 3.490347, 3.352302, 3.292738, 3.234742, 3.193368, 3.160927, 3.161128], 
[2.881099, 2.945246, 2.917418, 2.923599, 2.925177, 2.912500, 2.913650, 2.909926], 
[2.709178, 2.737958, 2.739958, 2.762623, 2.747041, 2.792302, 2.807778, 2.797556], 
[2.561972, 2.615291, 2.638952, 2.664163, 2.662676, 2.725002, 2.702721, 2.735946], 
[2.488111, 2.531158, 2.577694, 2.617544, 2.605485, 2.635562, 2.636641, 2.659380], 
[2.432198, 2.460672, 2.527182, 2.528783, 2.551888, 2.590004, 2.571233, 2.614528], 
[2.362630, 2.398171, 2.433268, 2.499655, 2.535710, 2.541973, 2.545331, 2.573424], 
[2.322370, 2.352458, 2.371212, 2.457805, 2.510648, 2.496917, 2.507852, 2.557567], 
[2.238095, 2.352366, 2.339051, 2.438003, 2.466364, 2.469078, 2.514749, 2.499165], 
[2.195452, 2.278896, 2.334433, 2.420428, 2.418045, 2.446898, 2.464002, 2.495093], 
[2.175775, 2.224480, 2.099380, 2.098904, 2.073245, 2.060422, 2.033604, 2.031129], 
[1.893682, 1.892100, 1.882532, 1.932661, 1.903853, 1.908416, 1.899865, 1.894059],
[1.827042, 1.836612, 1.844359, 1.861690, 1.862934, 1.822213, 1.900636, 1.902789]]
loss = np.mean(epochloss, axis=1)
x = np.linspace(1, 30, 30)
y = loss
plt.plot(x, y)
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.savefig("./Figures/loss_training.png")
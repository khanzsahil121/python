import numpy as np
import glob
import cv2
import math
import matplotlib.pyplot as plt
def lbpextract(b):
  # b = cv2.imread("0.jpg", 0)
  b = cv2.resize(b, (256, 256))
  b1 = np.zeros((258, 258))
  for ii in range(1, 257):
    for jj in range(1, 257):
      b1[ii, jj] = b[ii - 1, jj - 1]
  # print(b1)
  # print(b)
  # cv2.imshow('h',b)
  # cv2.destroyAllWindows()
  (a1, a2) = np.shape(b)
  (v1, v2) = np.shape(b1)
  k = 0
  r = np.zeros([256, 256])
  blocksize = 3;
  for ii in range(v1 - 2):
    for jj in range(v2 - 2):
      a11 = []
      k = 0
      w = []
      for i in range(blocksize):
        for j in range(blocksize):
          a11.append(b1[ii + i, jj + j])
          k = k + 1
      w = np.reshape(a11, (blocksize, blocksize))
      a11 = [a11[0], a11[1], a11[2], a11[5], a11[8], a11[7], a11[6], a11[3], a11[4]]
      # print(a11)
      # d=math.ceil(len(a11)/2)
      # print(d)
      f = []
      for i in range(len(a11) - 1):
        if a11[i] > a11[8]:
          # print(a11[i)
          f.append(1)
        else:
          f.append(0)
      # f.remove(f[d-1])
      # print(f)
      # j=int(f)
      int1 = 0;
      for k in range(8):
        int1 = int1 + (f[k] * (2 ** (k)))
      r[ii, jj] = int1
  # print(r)
  [aa, bb] = np.shape(r)
  # print(bb)
  cv2.imshow('ng',r)
  #cv2.waitKey()
  cv2.destroyAllWindows()
  hist, bin_edges = np.histogram(r, bins=16)
  plt.bar(bin_edges[:-1], hist, width=1)
  plt.xlim(min(bin_edges), max(bin_edges))
  plt.show()
  return hist

# from extract_lbp import *
path="E:\Sruthi\Ongoing Projects\Multichannel Decoded Local Binary Patterns for Content Based Image Retrieval\image.orig\*.jpg"
F=glob.glob(path)
for i in F:
   a1=cv2.imread(i)
   gray_image = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
   # a=cv2.resize(a1,(256,256))
   cv2.imshow('Window',gray_image)
   cv2.waitKey(1000)
   cv2.destroyAllWindows()
   f = lbpextract(gray_image)




















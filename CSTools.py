#!/usr/bin/env python
# -*- coding: <encoding name> -*-

""" 
CSTools.py: Collection of tools for cgn-data-21-1 
Capstone Project: Product Clustering

Functions: by kay Delventhal
    path_gen(name,PATH,silent=False)
    write_dict(name,data,PATH,silent=False)
    read_dict(name,PATH,silent=False)

Functions: Courtesy of ...
    centroid_histogram(clt):
    rgb2hsv(rgb)
    hsv2rgb(hsv):
    rgb_hue(rgb, s=1):
    rgb_saturation(rgb, s=1):
    rgb_value(rgb, s=1):
    rgb_hsv(rgb, s=1, dhue=1, dsat=1, dval=1):
    rgb_to_hsv(r, g, b):
    hsv_to_rgb(h, s, v):
    rgb_to_hsl(r, g, b):
    hsl_to_rgb(h, s, l):
    hsv_to_hsl(h, s, v):
    hsl_to_hsv(h, s, l):
"""

__author__  = "Kay Delventhal"
__license__ = "GPL"
__version__ = "0.1"
__status__  = "Development"

# import modules
import os
import pickle
import numpy as np
import math

# --- by kay Delventhal section ------------------------------------------------------

# to create project sub folders
def path_gen(name,PATH,silent=False):
    if len(name):
        path = PATH+name+'/'
    else:
        path = PATH
    if not os.path.isdir(path):
        os.makedirs(path)
        if not silent:
            print('makdir:', path)
    return path

# function to store python dict()
def write_dict(name,data,PATH,silent=False):
    path = PATH+name+'.pickle'
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=4)
        if not silent:
            print('write:', path)
        
# function to load python dict()
def read_dict(name,PATH,silent=False):
    path = PATH+name+'.pickle'
    if os.path.isfile(path):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
            if not silent:
                print('read:', path)
        return data
    else:
        print('error:', name,PATH,silent)
        return None

# --- "Courtesy of ..." section ----------------------------------------------
'''
https://www.easyrgb.com/en/convert.php#inputFORM
https://www.nixsensor.com/free-color-converter/
'''

# Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels, density=False)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist

# Courtesy of https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python
def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

# Courtesy of https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python
def hsv2rgb(hsv):
    """ convert HSV to RGB color space

    :param hsv: np.ndarray
    :return: np.ndarray
    """

    hi = np.floor(hsv[..., 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[..., 2].astype('float')
    f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
    p = v * (1.0 - hsv[..., 1])
    q = v * (1.0 - (f * hsv[..., 1]))
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

    return rgb

# Courtesy of https://gist.github.com/mathebox/e0805f72e7db3269ec22
def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

# Courtesy of https://gist.github.com/mathebox/e0805f72e7db3269ec22
def hsv_to_rgb(h, s, v):
    i = math.floor(h*6)
    f = h*6 - i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)

    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i%6)]

    return r, g, b

# Courtesy of https://curiousily.com/posts/color-palette-extraction-with-k-means-clustering/
def rgb_to_hex(rgb):
      return '#%s' % ''.join(('%02x' % p for p in rgb))

# Courtesy of https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
# see also http://cookbooks.adobe.com/post_Useful_color_equations__RGB_to_LAB_converter-14227.html
def rgb2lab( inputColor ) :
    
   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return L,a,b,Lab

# EOF
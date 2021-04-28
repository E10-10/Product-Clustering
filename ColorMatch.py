#!/usr/bin/env python
# -*- coding: <encoding name> -*-

""" 
CSData.py: Collection of tools for cgn-data-21-1 
Capstone Project: Product Clustering

Functions:
    KMeans_clustering(dict_begin,
                      images,
                      PATH,
                      DATA,
                      CLUSTER=4)
    return: -  -> saves data: ct.write_dict('img_hist_lst_'+str(CLUSTER),img_hist_lst,DATA)

    # fake gaussian distribution
    do_boost(p,boost):
    return: P

    gen_feature_vec(dict_begin,
                    img_hist_lst,
                    booster=BOOSTER,
                    ext='EXT',
                    scale=SCALE, 
                    chop=CHOP, 
                    stop=35000)
    return: data, stats

    dist_func_vec1(A, B)
    return: np.sqrt(sum((A - B)**2))

    dist_func_vec2(A, B)
    return: np.sqrt(np.sum(np.square(A - B)))

    dist_func_vec3(A1, B1, A2, B2, A3, B3)
    return: np.sqrt(np.sum(np.square(A1 - B1)) + 
                    np.sum(np.square(A2 - B2)) + 
                    np.sum(np.square(A3 - B3)))

    f_score_i(cl_real_i, cl_pred_i)
    return: 2*len(s_intsec) / (len(s_pred)+len(s_real))

    recall_i(cl_real_i, cl_pred_i)   
    return: (len(s_real) - len(s_diff_r_p)) / len(s_real) 

    precision_i(cl_real_i, cl_pred_i)
    return: (len(s_pred) - len(s_diff_p_r)) / len(s_pred)

    data_sort(all_data   = None,
              use        = 16,
              index      = 0,
              colorspace = 'lab')
    return: pid_lst_all, clr_lst_all, dst_lst_all

    # function to set up and prep the pre-clustered data
    # the pre-clustered data was feature engineered to 
    # provide vectors in RGB, HSV and LAB colorspace
    data_prep(all_data   = None,
              use        = 16,
              index      = 0,
             colorspace = 'lab')
    return: pid_lst_all, clr_lst_all

    calc_dist(pid_lst_ndx = None,
              pid_lbl_grp = None,
              pid_lst_all = None,
              clr_lst_all = None)
    return: dst_pid, plt_dat

    calc_scoring(dst_pid    = None,
                 dict_begin = None,
                 verbose    = False,
                 show       = 5,
                 knn_stop   = 10,
                 normalize  = True,
                 threshold  = 0.3,
                 norm_value = 100)
    return: goodness, f1scoreA, f1scoreB, recall, precision

    # (16, 2, 'lab', 20, 50, 0.5)
    calc_np_scoring(pid_lst  = None,
                    dst_lst    = None,
                    dict_begin = None,
                    verbose    = False,
                    show       = 10,
                    stop       = 100,
                    knn_stop   = 20,
                    normalize  = True,
                    threshold  = 0.5,
                    norm_value = 50)
    return: result
"""

__author__  = "Kay Delventhal"
__license__ = "GPL"
__version__ = "0.1"
__status__  = "Development"

# import modules
import os
from time import time
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import CSTools as ct

BOOSTER = 5
SCALE   = 1.0
CHOP    = (0,-1)

def KMeans_clustering(dict_begin,
                      images,
                      PATH,
                      DATA,
                      CLUSTER=4):
    """
    KMeans_clustering: function to cluster images by color 
                       with a given k
    input:
        dict_begin: project data as dict()
        images    : list() of images to cluster
        PATH      : project path
        DATA      :  project data path
        CLUSTER   : # of k-Means cluster to obtain
    output:
        none - file will be saved
        ct.write_dict('img_hist_lst_'+str(CLUSTER),img_hist_lst,DATA)
    """
    start = time()
    run = time()
    c = 0
    i = 0
    img_hist_lst = dict()
    stop = 35000
    for img, pid, lbl, target in zip(dict_begin['df_train_csv'].image.values, 
                             dict_begin['df_train_csv'].posting_id.values, 
                             dict_begin['df_train_csv'].label_group.values, 
                             dict_begin['df_train_csv'].target.values):
        file = PATH+'/'+img
        if file in images:
            img = os.path.basename(file)

            # load the image and convert it from BGR to RGB so that
            # we can dispaly it with matplotlib
            #width, height = 224, 224
            width, height = 175, 175
            dsize = (width, height)
            image = cv2.resize(cv2.imread(file),dsize)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # reshape the image to be a list of pixels
            image = image.reshape((image.shape[0] * image.shape[1], 3))

            # cluster the pixel intensities
            clt = KMeans(n_clusters=CLUSTER, random_state=42, verbose=0);
            clt.fit(image);

            hist = ct.centroid_histogram(clt)
            hist_img = list()
            for percent, color in zip(hist, clt.cluster_centers_):
                hist_img.append([percent, color])

            img_hist_lst[img] = [hist_img,hist, clt.cluster_centers_, [pid, lbl, target]]

            i += 1
            c += 1
            if not i % 50:
                print('.',end='')
            if i > 999:
                i = 0
                run = float((time()-run)/60)
                print(f'\ncount: {c} of {len(images)} rtime: {run:.2f}',end='')
                runs = float((time()-start)/60)
                left = float((run*33)-(run*(c/1000)))
                print(f' ttime: {runs:.2f} tleft: {left:.2f}',end='')
                run = time()
                ct.write_dict('img_hist_lst_temp',img_hist_lst,DATA,silent=True)

            stop -= 1
            if stop < 1:
                break

    ct.write_dict('img_hist_lst_'+str(CLUSTER),img_hist_lst,DATA)
    start = float((time()-start)/60)
    print(f'total time(m): {start:.2f}')
    start = float(start/60)
    print(f'total time(h): {start:.2f}')

# fake gaussian distribution
def do_boost(p,boost):
    """
    do_boost: function for a fake gaussian distribution
              to create a smeared (bigger) footprint
    input:
        p    : value to boost
        boost: # to chose width of boost
    output:
        P: list() with boosted p values
    """
    P = list()
    if int(boost) == 1:
        P.append(p)
    elif int(boost) == 2:
        P.append(p*0.95)
        P.append(p*0.95)
    elif int(boost) == 3:
        P.append(p*0.33)
        P.append(p*0.99)
        P.append(p*0.33)
    elif int(boost) == 4:
        P.append(p*0.4)
        P.append(p*0.95)
        P.append(p*0.95)
        P.append(p*0.4)
    elif int(boost) == 5:
        P.append(p*0.3)
        P.append(p*0.72)
        P.append(p*0.99)
        P.append(p*0.72)
        P.append(p*0.3)
    elif int(boost) == 6:
        P.append(p*0.1)
        P.append(p*0.25)
        P.append(p*0.76)
        P.append(p*0.99)
        P.append(p*0.76)
        P.append(p*0.25)
        P.append(p*0.1)
    elif int(boost) == 7:
        P.append(p*0.1)
        P.append(p*0.25)
        P.append(p*0.50)
        P.append(p*0.76)
        P.append(p*0.99)
        P.append(p*0.76)
        P.append(p*0.50)
        P.append(p*0.25)
        P.append(p*0.1)
    else:
        P.append(p*0.1)
        P.append(p*0.2)
        P.append(p*0.3)
        P.append(p*0.5)
        P.append(p*0.8)
        P.append(p*0.99)
        P.append(p*0.8)
        P.append(p*0.5)
        P.append(p*0.3)
        P.append(p*0.2)
        P.append(p*0.1)
    return P

def gen_feature_vec(dict_begin,
                    img_hist_lst,
                    booster=BOOSTER,
                    ext='EXT',
                    scale=SCALE, 
                    chop=CHOP, 
                    stop=35000):
    """
    do_boost: function for a fake gaussian distribution
              to create a smeared (bigger) footprint
    input:
        dict_begin  : project data as dict()
        img_hist_lst: k-Means cluster per image
        booster     : # of boost to use - see do_boost()
        ext         : not used anymore
        scale       : scale of p-value 
        chop        : can bu used to cut list() entries
        stop        : # for main loop to stop
    output:
        data : dict() with generated features
        stats: statistics for RGB, HSV and LAB range
    """

    print('img_hist_lst', len(img_hist_lst))
    print('posting_id  ', len(dict_begin['df_train_csv'].posting_id.values))
    print('BOOSTER ', booster)
    print('EXT     ', ext)
    print('SCALE   ', scale)
    print('CHOP    ', chop)
    print('stop    ', stop)

    data = dict()

    data['hsv'] = pd.DataFrame()
    data['rgb'] = pd.DataFrame()
    data['lab'] = pd.DataFrame() 

    hsv   = dict()
    rgb   = dict()    
    lab   = dict()
    
    def add_value(val_arry,p,value,boost_len,j,max_val,offset=0,clamp=True):
        pointer = int(round(value)) + int(offset) - int((boost_len/2)) + j
        if clamp and ((pointer < 0) or pointer > (max_val - 1)):
            pass
        else:
            if pointer > (max_val - 1):
                pointer -= max_val # + 1
            if pointer < 0:
                pointer = pointer + max_val # + 1
            val_arry[pointer] += p
    
    # loop through 'posting_id'
    counter = 0
    for img, pid, lbl, target in zip(dict_begin['df_train_csv'].image.values, 
                             dict_begin['df_train_csv'].posting_id.values, 
                             dict_begin['df_train_csv'].label_group.values, 
                             dict_begin['df_train_csv'].target.values):
        
        features_hsv = dict()
        features_rgb = dict()
        features_lab = dict()  
        
        # to obtain pd.DataFrame() header, strings are initialzied that also serve as an index
        hsv_h_max = 360
        hsv_s_max = 100
        hsv_v_max = 100
        hsv_h = np.zeros(hsv_h_max)
        hsv_s = np.zeros(hsv_s_max)
        hsv_v = np.zeros(hsv_v_max)
        
        # to obtain pd.DataFrame() header, strings are initialzied that also serve as an index
        lab_l_max = 100
        lab_a_max = 2*100
        lab_b_max = 2*100
        lab_l = np.zeros(lab_l_max)
        lab_a = np.zeros(lab_a_max)
        lab_b = np.zeros(lab_b_max)
        
        # to obtain pd.DataFrame() header, strings are initialzied that also serve as an index
        rgb_r_max = 255+1 # 0-255
        rgb_g_max = 25+1 # 0-255
        rgb_b_max = 25+1 # 0-255
        rgb_r = np.zeros(rgb_r_max)
        rgb_g = np.zeros(rgb_g_max)
        rgb_b = np.zeros(rgb_b_max)
        
        # get Kmeans color cluster list, which is #-cluster*[p, (r,g,b)] long 
        label_prgb = img_hist_lst[img][0]
        try:
            label_prgb.sort(reverse=True)
        except Exception as err:
            try:
                label_prgb = sorted(label_prgb, key=lambda x: x[0])
            except Exception as err:
                print(img, pid, lbl, target)
                print(label_prgb)
                print('ERR:', err)
        label_prgb = label_prgb[chop[0]:chop[-1]]
        
        stats = dict()
        stats['R'],  stats['G'],  stats['B']  = set(), set(), set()
        stats['H'],  stats['S'],  stats['V']  = set(), set(), set()
        stats['L_'], stats['A_'], stats['B_'] = set(), set(), set()
        
        # loop through sorted color clusters: big(p) -> small(p)
        rgb_lst = list()
        for i, each in enumerate(label_prgb):
            rgb_lst.append(each)
            p, (r,g,b) = each
            if r < 0: r = 0.
            if g < 0: g = 0.
            if b < 0: b = 0.
            h, s, v     = ct.rgb_to_hsv(r,g,b)
            l_,a_,b_,_  = ct.rgb2lab([r,g,b])
            
            stats['R'].add(int(r))
            stats['G'].add(int(g))
            stats['B'].add(int(b))
            stats['H'].add(int(h))
            stats['S'].add(int(s))
            stats['V'].add(int(v))
            stats['L_'].add(int(l_))
            stats['A_'].add(int(a_))
            stats['B_'].add(int(b_))

            try:
                # the 'BOOSTER' will widen the footprint of the data
                # instate of adding one value only we add a gaussian distribution of values
                p_boosted = do_boost(p * scale, booster)
                
                for j, p_ in enumerate(p_boosted):
                    # 'h' is the hue of the hsv-color that is a value number between 0/360 
                    add_value(hsv_h,p_,h,len(p_boosted),j,hsv_h_max,clamp=False)
                          
                    # 's' is the stauration of the hsv-color that is a number between 0/100
                    add_value(hsv_s,p_,s,len(p_boosted),j,hsv_s_max)
                          
                    # 'v' is the value of the hsv-color that is a number between 0/100
                    add_value(hsv_v,p_,v,len(p_boosted),j,hsv_v_max)

                    # 'l_' is the value of the lab-color that is a number between 0/100
                    add_value(lab_l,p_,l_,len(p_boosted),j,lab_l_max)
                          
                    # 'a_' is the value of the lab-color that is a number between -128/128
                    add_value(lab_a,p_,a_,len(p_boosted),j,lab_a_max,offset=int(lab_a_max/2))
                          
                    # 'b_' is the value of the lab-color that is a number between -128/128
                    add_value(lab_b,p_,b_,len(p_boosted),j,lab_l_max,offset=int(lab_b_max/2))

                    # 
                    add_value(rgb_r,p_,r,len(p_boosted),j,rgb_r_max)
                          
                    # 
                    add_value(rgb_g,p_,g,len(p_boosted),j,rgb_g_max)
                          
                    # 
                    add_value(rgb_b,p_,b,len(p_boosted),j,rgb_b_max)
            
            except Exception as err:
                print('-> ERR', err)
                print(img, pid, lbl, target)
                print('p,r,g,b :', p,r,g,b)
                print('h,s,v   :', h,s,v)
                print('l_,a_,b_:', l_,a_,b_)
                break
                
        if stop < 1:
            break
        stop -= 1

        counter += 1
        # build up dict() - for df_hsv
        features_hsv.update({'pid':pid})
        features_hsv.update({'img':img})
        features_hsv.update({'lbl':str(lbl)})
        features_hsv.update({'target':target})
        features_hsv.update({'hsv_h':hsv_h})
        features_hsv.update({'hsv_s':hsv_s})
        features_hsv.update({'hsv_v':hsv_v})
        # convert dict() into pd.DataFrame()
        data['hsv'] =data['hsv'].append(features_hsv, ignore_index=True)

        # build up dict() - for df_hsv
        features_rgb.update({'pid':pid})
        features_rgb.update({'img':img})
        features_rgb.update({'lbl':str(lbl)})
        features_rgb.update({'target':target})
        features_rgb.update({'rgb_r':rgb_r})
        features_rgb.update({'rgb_g':rgb_g})
        features_rgb.update({'rgb_b':rgb_b})
        # convert dict() into pd.DataFrame()
        data['rgb'] =data['rgb'].append(features_rgb, ignore_index=True)

        # build up dict() - for df_hsv
        features_lab.update({'pid':pid})
        features_lab.update({'img':img})
        features_lab.update({'lbl':str(lbl)})
        features_lab.update({'target':target})
        features_lab.update({'lab_l':lab_l})
        features_lab.update({'lab_a':lab_a})
        features_lab.update({'lab_b':lab_b})
        # convert dict() into pd.DataFrame()
        data['lab'] =data['lab'].append(features_lab, ignore_index=True)
        
    for each in stats.keys():
        print(each,max(stats[each]),min(stats[each]))
          
    print('counter', counter)
    return data, stats

def dist_func_vec1(A, B):
    """
    dist_func_vec1: calculate euclidian distance
    input:
        A:vector
        B: vector
    output:
        distance
    """
    return np.sqrt(sum((A - B)**2))

def dist_func_vec2(A, B):
    """
    dist_func_vec1: calculate euclidian distance
    input:
        A:vector
        B: vector
    output:
        distance
    """
    return np.sqrt(np.sum(np.square(A - B)))

def dist_func_vec3(A1, B1, A2, B2, A3, B3):
    """
    dist_func_vec1: calculate euclidian distance
    input:
        A:vector
        B: vector
    output:
        distance
    """
    return np.sqrt(np.sum(np.square(A1 - B1)) + np.sum(np.square(A2 - B2)) + np.sum(np.square(A3 - B3)))

def data_sort(all_data   = None,
              use        = 16,
              index      = 0,
              colorspace = 'lab'):
    """
    data_prep: function optimized for speed with np.array()
    input:
        all_data  : dict() with raw data
        use       : to chose a certain data set
        index     : to chose a version in that data set
        colorspace: to chose colorspace to use
    output:
        pid_lst_all: np.array() with all pids
        clr_lst_all: np.array() with all color vectors
        dst_lst_all: np.array() 34250x34250 with all distnaces
    """
    keys = list(all_data[use].keys())
    if colorspace == 'lab':
        df = all_data[use][keys[index]][0]['lab'] 
        cs = ['lab_l','lab_a','lab_b']
    elif colorspace == 'hsv':
        df = all_data[use][keys[index]][0]['hsv'] 
        cs = ['hsv_h','hsv_s','hsv_v']
    else:
        df = all_data[use][keys[index]][0]['rgb'] 
        cs = ['rgb_r','rgb_g','rgb_b']

    pid_lst_all = df['pid'].to_numpy()
    clr_lst_all = df[cs].to_numpy()
    ln_ = df.shape[0]
    dst_lst_all = np.zeros((ln_,ln_))
    del df
    del all_data
    
    i_c = 0
    j_c = 0
    for i, pid in enumerate(pid_lst_all):
        i_c += 1
        for j, clr in enumerate(clr_lst_all):
            j_c += 1
            dst_lst_all[i][j] = dist_func_vec3(clr_lst_all[i][0], # A1
                                               clr_lst_all[j][0], # B1
                                               clr_lst_all[i][1], # A2
                                               clr_lst_all[j][1], # B2
                                               clr_lst_all[i][2], # A3
                                               clr_lst_all[j][2]) # B3
            #dist = cm.dist_func_vec3(A1, B1, A2, B2, A3, B3)
            if j_c > 5000:
                print('j',end='')
                j_c = 0
        if i_c > 5000:
            print('i')
            i_c = 0
    #dst_lst_all.sort(axis=1)
    #a.view('i8,i8,i8').sort(order=['f1'], axis=0)
    return pid_lst_all, clr_lst_all, dst_lst_all

# function to set up and prep the pre-clustred data
# the pre-clustered data was feature engineered to 
# provide vectors in RGB, HSV and LAB colorspace

def data_prep(all_data   = None,
              use        = 16,
              index      = 0,
              colorspace = 'lab'):
    """
    data_prep: function to set up and prep the pre-clustred data
               the pre-clustered data was feature engineered to 
               provide vectors in RGB, HSV and LAB colorspace
    input:
        all_data  : dict() with raw data
        use       : to chose a certain data set
        index     : to chose a version in that data set
        colorspace: to chose colorspace to use
    output:
        pid_lst_all: np.array() with all pids
        clr_lst_all: np.array() with all color vectors
    """
    keys = list(all_data[use].keys())
    df_rgb = all_data[use][keys[index]][0]['rgb'] 
    df_hsv = all_data[use][keys[index]][0]['hsv'] 
    df_lab = all_data[use][keys[index]][0]['lab'] 
        
    if colorspace == 'lab':
        df = df_lab
        cs = ['lab_l','lab_a','lab_b']
    elif colorspace == 'hsv':
        df = df_hsv
        cs = ['hsv_h','hsv_s','hsv_v']
    else:
        df = df_rgb
        cs = ['rgb_r','rgb_g','rgb_b']

    pid_lst_all = df['pid'].to_numpy()
    clr_lst_all = df[cs].to_numpy()    
    return pid_lst_all, clr_lst_all

def calc_dist(pid_lst_ndx = None,
              pid_lbl_grp = None,
              pid_lst_all = None,
              clr_lst_all = None):
    """
    calc_dist: function to calculate the color diffrence in 
               in a colorspace ( RGB, HSV or LAB)
    input:
        pid_lst_ndx: list() with pid indices 
        pid_lbl_grp: list() with lable_groups by pids
        pid_lst_all: np.array() with all pids
        clr_lst_all: np.array() with all color vectors
    output:
        dst_pid: sorted list() with: dist,lbl_grp,pid_j
        plt_dat: sorted list() with distances for ploting
    """
    dst_pid = dict()
    plt_dat = dict()
    for i in tqdm(pid_lst_ndx):
        pid_s = pid_lst_all[i]
        dst_pid[pid_s] = list()
        plt_dat[pid_s] = list()
        dst_lst = list()
        all_lst = list()
        for j, clr_j in enumerate(clr_lst_all):
            pid_j = pid_lst_all[j]
            A1 = clr_lst_all[i][0]
            A2 = clr_lst_all[i][1]
            A3 = clr_lst_all[i][2]
            B1 = clr_j[0]
            B2 = clr_j[1]
            B3 = clr_j[2]
            dist = dist_func_vec3(A1, B1, A2, B2, A3, B3)
            dst_lst.append(dist)
            lbl_grp = pid_lbl_grp[pid_j][0]
            all_lst.append([dist,lbl_grp,pid_j])
        dst_pid[pid_s] = sorted(all_lst,reverse=False)
        plt_dat[pid_s] = sorted(dst_lst,reverse=False)
    return dst_pid, plt_dat

def calc_scoring(dst_pid    = None,
                 dict_begin = None,
                 verbose    = False,
                 show       = 5,
                 knn_stop   = 10,
                 normalize  = True,
                 threshold  = 0.3,
                 norm_value = 100):
    """
    calc_scoring: function to calculate scores 
                  for a set of distances
    input:
        dst_pid   : dict() per pid with distances
        dict_begin: project data as dict()
        verbose   : if True prints results
        show      : # of results printed
        knn_stop  : hard limited for cluster size
        normalize : if True distances will be normalized
        threshold : threshold for clustering
        norm_value: index for curve threshold
    output:
        goodness : list() of self defined score
        f1scoreA : list() of tuned f1-score
        f1scoreB : list() of real f1-score
        recall   : list() of recall
        precision: list() of precision
    """
    goodness = list()
    f1scoreA = list()
    f1scoreB = list()
    recall   = list()
    precision= list()
    for pid_s in dst_pid:
        delta = 0.
        last = 0.
        lbl_grp = dict_begin['dic_posting_id_label_group'][pid_s][0]
        lbL_grp_len = len(dict_begin['dic_label_group_posting_id'][lbl_grp])
        if verbose:
            print('\n->', f'{pid_s:10s} / {lbL_grp_len:2d} /', lbl_grp)

        count_hit = 0
        counter   = 1
        norm_val  = dst_pid[pid_s][norm_value][0]
        for d in dst_pid[pid_s]:
            dist  = d[0]
            delta = dist - last
            if d[1] == lbl_grp:
                hit = 'XX'
                count_hit += 1
            if normalize:
                dist = dist / norm_val
            if verbose and counter < show:
                print(f'{dist:.5f}  {str(d[1]):10s}  {str(d[2]):18s} {hit} {(count_hit/lbL_grp_len):.3f}')
            if counter > knn_stop:
                break
            if dist > threshold:
                break
            last = dist
            hit = '  '
            counter += 1

        goodness.append(count_hit/lbL_grp_len)
        f1scoreA.append(2*count_hit/(lbL_grp_len+count_hit))
        f1scoreB.append(2*count_hit/(lbL_grp_len+counter))  
        recall.append( (lbL_grp_len - (lbL_grp_len-count_hit)) / lbL_grp_len  )
        precision.append( (counter - (counter-count_hit)) / counter )
    return goodness, f1scoreA, f1scoreB, recall, precision

# (16, 2, 'lab', 20, 50, 0.5)
def calc_np_scoring(pid_lst  = None,
                    dst_lst    = None,
                    dict_begin = None,
                    verbose    = False,
                    show       = 10,
                    stop       = 100,
                    knn_stop   = 20,
                    normalize  = True,
                    threshold  = 0.5,
                    norm_value = 50):
    """
    calc_np_scoring: calc_np_scoring() is based on calc_scoring
                     but it is optimized for speed with np.array()
    input:
        pid_lst   : np.array() with pid
        dst_lst   : np.array() with distances
        dict_begin: project data as dict()
        verbose   : if True prints results
        show      : # of results printed
        knn_stop  : hard limited for cluster size
        normalize : if True distances will be normalized
        threshold : threshold for clustering
        norm_value: index for curve threshold
    output:
        result: dict() by pid_i with (dist,pid_d)
    """    
    result = dict()  
    for i, pid_i in enumerate(pid_lst):
        stop -= 1
        if stop < 0:
            break
        dists = np.dstack((dst_lst[i],pid_lst))[0]
        
        try:
            dists.sort(reverse=True)
            #b = dst_lst[1].sort(axis=0)
            #a = dst_lst[0]
        except Exception as err:
            try:
                dists = sorted(dists, key=lambda x: x[0])
            except Exception as err:
                print('ERR:', err)                
                
        lbl_grp_i = dict_begin['dic_posting_id_label_group'][pid_i][0]
        lbL_grp_i_len = len(dict_begin['dic_label_group_posting_id'][lbl_grp_i])

        res_lst = list()
        count_hit = 0
        counter   = 1
        norm_val  = dists[norm_value][0] 
        if verbose:
            print('\n->', f'{i} {pid_i:10s} / {lbL_grp_i_len:2d} / {lbl_grp_i}', norm_val, threshold)
        for dist, pid_d in dists:
            lbl_grp_d = dict_begin['dic_posting_id_label_group'][pid_d][0]
            if normalize:
                dist = dist / norm_val
            if dist >= threshold:
                break
            res_lst.append((dist,pid_d))
            if str(lbl_grp_d) == str(lbl_grp_i):
                hit = 'XX'
                count_hit += 1
            else:
                hit = '  '
            if verbose and counter < show:
                print(f'{dist:.6f} {str(pid_d):18s} {str(lbl_grp_d):18s} {hit} {(count_hit/lbL_grp_i_len):.3f}')
            if counter > knn_stop:
                break
            counter += 1
        result[pid_i] = res_lst
    return result

# EOF
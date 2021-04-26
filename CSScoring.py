#!/usr/bin/env python
# -*- coding: <encoding name> -*-

""" 
CSScoring.py: Collection of tools for cgn-data-21-1 
Capstone Project: Product Clustering
"""

__author__  = "Elias Büchner / Niels-Christian Leight"
__license__ = "GPL"
__version__ = "0.1"
__status__  = "Development"

# import modules
import os
import pickle
import numpy as np
import pandas as pd

def f_score_i(cl_real_i, cl_pred_i):
    '''
    Description:
    Calculate f-score for a single posting_id
    f1-score is the mean of all f-scores
    Parameters: 
    argument1 (list): list of posting_id's belonging to the real cluster
    argument2 (list): list of posting_id's belonging to the predicted cluster
    Returns: 
    float value of f-score   
    '''
    s_pred = set(cl_pred_i)
    s_real = set(cl_real_i)
    s_intsec = s_pred.intersection(s_real)
    return 2*len(s_intsec) / (len(s_pred)+len(s_real))

def recall_i(cl_real_i, cl_pred_i):      
    '''
    Description:
    Calculate recall for a single posting_id
    Parameters: 
    argument1 (list): list of posting_id's belonging to the real cluster
    argument2 (list): list of posting_id's belonging to the predicted cluster
    Returns: 
    float value of recall   
    '''
    s_pred = set(cl_pred_i)
    s_real = set(cl_real_i)   
    s_diff_r_p = s_real.difference(s_pred)
    return (len(s_real) - len(s_diff_r_p)) / len(s_real) 

def precision_i(cl_real_i, cl_pred_i):      
    '''
    Description:
    Calculate precision for a single posting_id
    Parameters: 
    argument1 (list): list of posting_id's belonging to the real cluster
    argument2 (list): list of posting_id's belonging to the predicted cluster
    Returns: 
    float value of precision   
    '''
    s_pred = set(cl_pred_i)
    s_real = set(cl_real_i)    
    s_diff_p_r = s_pred.difference(s_real)
    return (len(s_pred) - len(s_diff_p_r)) / len(s_pred)

# Define a function that return all images which actually belong to the cluster of a certain image i
# The result is returned as a list containing strings

def get_sim_all_pi(i_vec_1,i_vec_all):  
    return i_vec_all.dot(i_vec_1)


def get_sim_two_pi(i_vec_1,i_vec_2):
    sim = np.dot(i_vec_1,i_vec_2)/(np.linalg.norm(i_vec_1)*np.linalg.norm(i_vec_2))
    return sim

def dist2(x,y,label_vec):          # x,y indicate the position of the two images in our DataFrame
    a = label_vec[x]
    b = label_vec[y]
    dist = np.sqrt(sum([(a[i] - b[i])**2 for i in range(label_vec.shape[1])]))        # (Euclidean Metric)
    #dist = sum([abs((a[i] - b[i])) for i in range(label_vec.shape[1])])              # (Manhattan-Metric)
    
    return dist

def real_cluster_of_i_w2v(i,df):
    '''
    Description:
    Find real cluster for a single posting_id
    Use this function when working with Word2Vec
    
    Parameters: 
    argument1 (int): position of posting_id in DataFrame
    
    Returns: 
    list of all posting_id's  
    '''
    
    l_g = (df.iloc[i].at['label_group'])
    df_red = df[df['label_group'] == l_g]
    df_red_list = df_red['posting_id'].tolist()
    return df_red_list

def pred_cluster_of_i_w2v(i,threshold,df,labels,posting_id):
    '''
    Description:
    Find predicted cluster for a single posting_id
    Use this function when working with Word2Vec
    
    Parameters: 
    argument1 (int): position of posting_id in DataFrame
    
    Returns: 
    list of all posting_id's  
    '''

    list1 = []
    list2 = []
    list3 = []
        
    for j in range(34250):

        i_vec_1 = df['word_vec'][j]
        i_vec_2 = df['word_vec'][i]
        list1.append(round(get_sim_two_pi(i_vec_1, i_vec_2),4))
        
        list2.append(labels[j])
        list3.append(posting_id[j])
                
    df_nlp = pd.DataFrame(data = [list1,list2,list3]).transpose()
    df_nlp = df_nlp.sort_values(by = 0)
    
    df_nlp = df_nlp[df_nlp[0] >= threshold]
    
    ls = df_nlp[2].tolist()
    
    return ls

# EOF
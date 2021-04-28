#!/usr/bin/env python
# -*- coding: <encoding name> -*-

""" 
CSData.py: Collection of tools for cgn-data-21-1 
Capstone Project: Product Clustering

Functions:
    my_setup(work_path              ='/output',
             input_path_train_image ='/shopee-product-matching/train_images' , 
             input_path_test_image  ='/shopee-product-matching/test_images' ,
             input_path_train_csv   ='/shopee-product-matching/train.csv',
             input_path_test_csv    ='/shopee-product-matching/test.csv',
             input_path_sasu_csv    ='/shopee-product-matching/sample_submission.csv',
             n_of_label_groups      =LBLGRP,
             RSEED = 42)
    return: dict_setup

    my_begin(dict_setup)
    return: dict_begin
"""

__author__  = "Elias Büchner"
__license__ = "GPL"
__version__ = "0.1"
__status__  = "Development"

# import modules
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LBLGRP = 11014
RSEED = 42

def my_setup(work_path             ='/output',
             input_path_train_image='/shopee-product-matching/train_images' , 
             input_path_test_image ='/shopee-product-matching/test_images' ,
             input_path_train_csv  ='/shopee-product-matching/train.csv',
             input_path_test_csv   ='/shopee-product-matching/test.csv',
             input_path_sasu_csv   ='/shopee-product-matching/sample_submission.csv',
             n_of_label_groups     =LBLGRP,
             RSEED                 =RSEED):
    
    # change  varible
    dict_setup={'work_path'             :work_path,
                'input_path_train_image':input_path_train_image,
                'input_path_test_image' :input_path_test_image,
                'input_path_train_csv'  :input_path_train_csv,
                'input_path_test_csv'   :input_path_test_csv,
                'input_path_sasu_csv'   :input_path_test_csv,
                'n_of_label_groups'     :n_of_label_groups,
                'RSEED':RSEED}
    return dict_setup

def my_begin(dict_setup):
    # load csv data
    df_train_all = pd.read_csv(dict_setup['input_path_train_csv'])
    df_test      = pd.read_csv(dict_setup['input_path_test_csv'])
    df_sasu      = pd.read_csv(dict_setup['input_path_sasu_csv'], index_col = 0) #sample_submission

    train_images = dict_setup['input_path_train_image'] + '/' + df_train_all['image']
    df_train_all['path'] = train_images

    test_images     = dict_setup['input_path_test_image'] + '/' +df_test['image']
    df_test['path'] = test_images

    # cut data frame, to have just N_of_all_label_groups
    N_of_label_groups = dict_setup['n_of_label_groups']
    if N_of_label_groups == LBLGRP:
        # all label_groups
        df_train = df_train_all
    else:
        ls_label_groups = list(df_train_all.label_group.unique())        
        N_of_all_label_groups = len(ls_label_groups)    
        RSEED = dict_setup['RSEED']
        df_ratio = N_of_label_groups/N_of_all_label_groups
        lg_want, lg_rest = train_test_split(ls_label_groups,train_size=df_ratio, random_state=RSEED) 
        df_train = df_train_all[df_train_all["label_group"].isin( lg_want )]

    # add target
    # create dictionary key label_group; value list posting_id
    dic_label_group_posting_id = df_train.groupby('label_group').posting_id.agg('unique').to_dict()
    # create dictionary ....
    dic_label_group_image = df_train.groupby('label_group').image.agg('unique').to_dict()
    # create dictionary ....
    dic_posting_id_image = df_train.groupby('posting_id').image.agg('unique').to_dict()
    # create dictionary ....
    dic_posting_id_image_phash = df_train.groupby('posting_id').image_phash.agg('unique').to_dict()
    # create dictionary ....
    dic_posting_id_label_group = df_train.groupby('posting_id').label_group.agg('unique').to_dict()
    # create dictionary ....
    dic_image_label_group = df_train.groupby('image').label_group.agg('unique').to_dict()
    # create dictionary ....
    dic_image_posting_id = df_train.groupby('image').posting_id.agg('unique').to_dict()
    # map dict in new column target for given dict above
    df_train['target'] = df_train.label_group.map(dic_label_group_posting_id)
    
    # dict
    dict_begin ={'dict_setup' : dict_setup,
                'df_train_csv': df_train,
                'df_test_csv' : df_test,
                'df_sasu_csv' : df_sasu,
                'dic_label_group_posting_id':dic_label_group_posting_id,
                'dic_posting_id_image_phash':dic_posting_id_image_phash,
                'dic_posting_id_image'      :dic_posting_id_image,
                'dic_posting_id_label_group':dic_posting_id_label_group,
                'dic_image_label_group'     :dic_image_label_group,
                'dic_label_group_image'     :dic_label_group_image,
                'dic_image_posting_id'      :dic_image_posting_id}
    
    return dict_begin

# EOF
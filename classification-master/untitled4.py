# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:43:08 2020

@author: lab1016
"""
import os
from untitled3 import SearchResults

temp_dir = os.getcwd()
res = SearchResults(directory=temp_dir+'saved_models3',
                    project_name='cifar10_%s_model.087.h5',
                    objective='val_acc'                    
                   )
res.reload()
model = res.get_best_models()[0]
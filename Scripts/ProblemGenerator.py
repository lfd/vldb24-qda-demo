#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import json
import os
import pathlib 
import pickle
from decimal import *


# In[ ]:


def get_rounded_val(val, num_decimal_pos=4):
    return Decimal(val).quantize(Decimal(10) ** -num_decimal_pos, rounding=ROUND_HALF_EVEN)

def parse_selectivities(sel):
    pred = []
    pred_sel = []
    for (i, j) in itertools.combinations(range(len(sel)), 2):
        if sel[i][j] != 1:
            pred.append((i, j))
            pred_sel.append(sel[i][j])
    return pred, pred_sel

def format_loaded_pred(pred):
    form_pred = []
    for p in pred:
        form_pred.append(tuple(p))
    return form_pred


# In[2]:


def get_join_ordering_qubo(problempath):
    
    rd = os.path.abspath(problempath)
    pathlib.Path(rd).mkdir(parents=True, exist_ok=True) 
    
    qubofile = os.path.abspath(problempath + "/qubo.txt")
    qubo = None
    with open(qubofile, 'rb') as file:
        qubo = pickle.load(file)
    return qubo

def save_join_ordering_qubo(problempath, qubo):
    rd = os.path.abspath(problempath)
    pathlib.Path(rd).mkdir(parents=True, exist_ok=True) 
    
    qubofile = os.path.abspath(problempath + "/qubo.txt")
    with open(qubofile, 'wb') as file:
        qubo = pickle.dump(qubo, file)
        
# generated_problems indicates whether or not the problems have been generated
# using the query generation code by Trummer (src: https://github.com/itrummer/query-optimizer-lib)
def get_join_ordering_problem(problem_path, generated_problems=True):
    if generated_problems:
        card = load_from_path(problem_path + '/cardinalities.json')
        sel = load_from_path(problem_path + '/selectivities.json')
        pred, pred_sel = parse_selectivities(sel)
        return card, pred, pred_sel
    else:
        card = load_from_path(problem_path + "/card.txt")      
        pred = format_loaded_pred(load_from_path(problem_path + "/pred.txt"))    
        pred_sel = load_from_path(problem_path + "/pred_sel.txt")
        return card, pred, pred_sel

def load_from_path(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file) as file:
            data = json.load(file)
            return data

def load_join_ordering_problem(problem_path):

    card_file = 'cardinalities.json'
    sel_file = 'selectivities.json'
    
    card_path = os.path.abspath(problem_path + '/' + card_file)
    card = []
    with open(card_path, 'r') as file:
        card = json.load(file)
    
    sel_path = os.path.abspath(problem_path + '/' + sel_file)
    sel = []
    with open(sel_path, 'r') as file:
        sel = json.load(file)
    
    return card, sel


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import prod
import math
import itertools
from math import inf
from sympy.utilities.iterables import multiset_permutations

from os import listdir
from os.path import isfile, join
from pathlib import Path
import time
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


def get_selectivity_for_new_relation(join_order, j, pred, pred_sel):
    sel = 1
    new_relation = join_order[j]
    for i in range(j):
        relation = join_order[i]
        if (relation, new_relation) in pred:
            sel = sel * pred_sel[pred.index((relation, new_relation))]
        elif (new_relation, relation) in pred:
            sel = sel * pred_sel[pred.index((new_relation, relation))]
    return sel

def get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=False):
    int_costs = []
    join_order = join_order.copy()
    if join_order[0] > join_order[1]:
        join_order[0], join_order[1] = join_order[1], join_order[0]
    prev_join_result = card[join_order[0]]
    for j in range(1, len(card)-1):
        jo_hash = str(join_order[0:j+1])
        if jo_hash in card_dict:
            int_card = card_dict[jo_hash]
        else:
            sel = get_selectivity_for_new_relation(join_order, j, pred, pred_sel)
            int_card = prev_join_result * card[join_order[j]] * sel
            card_dict[jo_hash] = int_card
        prev_join_result = int_card
        int_costs.append(int_card)
    if verbose:
        print(int_costs)
    return int_costs

def calculate_cost_for_join(relations, card, pred, pred_sel):
    cost = 0
    card_prod = np.prod(np.array(card)[relations])
    pred_prod = 1
    for p in range(len(pred)):
        (r1, r2) = pred[p]
        if r1 in relations and r2 in relations:
            pred_prod = pred_prod * pred_sel[p]
    cost = card_prod * pred_prod
    return cost

def get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, card_dict, verbose=False):
    total_costs = 0 
    int_costs = get_intermediate_costs_for_join_order(join_order, card, pred, pred_sel, card_dict, verbose=verbose)
    for cost in int_costs:
        total_costs = total_costs + cost
    return total_costs

def get_costs_for_bushy_tree(join_list, card, pred, pred_sel):
    cost = 0
    for relations in join_list:
        if len(relations) < 2:
            continue
        if len(relations) == len(card):
            continue
        card_prod = np.prod(np.array(card)[relations])
        pred_prod = 1
        for p in range(len(pred)):
            (r1, r2) = pred[p]
            if r1 in relations and r2 in relations:
                pred_prod = pred_prod * pred_sel[p]
        intermediate_cost = card_prod * pred_prod
        cost = cost + intermediate_cost
    return cost


# In[ ]:


def get_raw_join_order(cost_vector):
    join_order = np.argsort(cost_vector).tolist()
    join_order.reverse()
    return join_order

def postprocess_join_order(raw_join_order, cost_vector, num_relations, pred):
    join_order = [raw_join_order[0]]
    while len(join_order) < num_relations:
        
        applicable_predicates = [pred_tuple for t in join_order for pred_tuple in pred if t in pred_tuple]
        neighborhood_indices = [t for t in set(sum(applicable_predicates, ())) if t not in join_order]
        
        if len(neighborhood_indices) != 0:
            best_neighbor_relation = neighborhood_indices[np.argmax(cost_vector[neighborhood_indices])]
            join_order.append(best_neighbor_relation)
        else:
            global_indices = [x for x in raw_join_order if x not in join_order]
            best_global_relation = global_indices[np.argmax(cost_vector[global_indices])]
            join_order.append(best_global_relation)
    return join_order
    
def display_heatmap(array):
    fig, ax = plt.subplots()
    sns.heatmap(array, ax=ax)
    st.pyplot(fig)
    
def get_ideal_bitstring(num_relations):
    num_joins = num_relations - 2
    ideal_bitstrings = []
    ideal_bitstrings.append(np.ones(num_joins).tolist())
    for i in range(num_relations-1):
        joins = []
        ones = np.ones(num_joins-i).tolist()
        zeros = np.zeros(num_joins-len(ones)).tolist()
        joins.extend(zeros)
        joins.extend(ones)
        ideal_bitstrings.append(joins)
    return ideal_bitstrings
    
def contains_errors(cost_vector, weight_vector, num_relations):
    ideal_bitstrings = get_ideal_bitstring(num_relations)
    ideal_cost_vector = np.array(ideal_bitstrings).dot(weight_vector).tolist()
    for cost_value in cost_vector:
        if cost_value in ideal_cost_vector:
            ideal_cost_vector.remove(cost_value)
    if len(ideal_cost_vector) == 0:
        return False
    return True
    
def get_bitstrings_for_join_order(join_order):
    num_relations = len(join_order)
    num_joins = num_relations-2
    bitstrings = np.empty((num_relations, num_joins))
    bitstrings[join_order[0]] = np.ones(num_joins).tolist()
    for i in range(num_relations-1):
        relation = join_order[i+1]
        joins = []
        ones = np.ones(num_joins-i).tolist()
        zeros = np.zeros(num_joins-len(ones)).tolist()
        joins.extend(zeros)
        joins.extend(ones)
        bitstrings[relation] = np.array(joins)
    return bitstrings
    
def generate_difference_matrix(initial_bitstrings, repaired_bitstrings, num_relations):
    num_joins = num_relations - 2
    difference_matrix = np.empty((num_relations, num_joins))
    for i in range(num_relations):
        for j in range(num_joins):
            if initial_bitstrings[i][j] == 0 and repaired_bitstrings[i][j] == 1:
                difference_matrix[i][j] = 0.9
            elif initial_bitstrings[i][j] == 1 and repaired_bitstrings[i][j] == 0:
                difference_matrix[i][j] = 0.1
            else: 
                difference_matrix[i][j] = initial_bitstrings[i][j]
    return difference_matrix
   
def readout(response, card, pred, pred_sel, card_dict, visualise_process):
    start = time.time()
    bitstrings = []
    for solution in response[0]:
        for i in range(int(solution[1])):
            bitstrings.append(solution[0])
            
            
    weight_vector = np.arange(1, len(card)-1)
    weight_vector = weight_vector[len(card)-3::-1]
    
    best_solutions_for_time = []
    best_costs = inf
        
    solutions = []
    
    num_relations = len(card)
    
    for i in range(len(bitstrings)):
        bitstring = bitstrings[i]
        
        
        bitstring = bitstring[:len(card)*(len(card)-2)]
        partial_bitstrings = np.array_split(bitstring, len(card))
        if visualise_process:
            with st.expander("Annealing Result " + str(i+1)):
                display_heatmap(partial_bitstrings.copy())
        cost_vector = np.array(partial_bitstrings).dot(weight_vector)
        
        raw_join_order = get_raw_join_order(cost_vector)
        
        if visualise_process:
            st.write("Validating Annealing Solution...")
            if contains_errors(cost_vector.copy().tolist(), weight_vector, num_relations):
                st.write("Bit Errors Detected!")
                st.write("Repairing Annealing Result...")
                repaired_bitstrings = get_bitstrings_for_join_order(raw_join_order.copy())
                st.write("Successfully Repaired!")
                difference_matrix = generate_difference_matrix(partial_bitstrings.copy(), repaired_bitstrings.copy(), num_relations)
                with st.expander("Repaired Annealing Result"):
                    display_heatmap(difference_matrix)
            else:
                st.write("No Errors Detected!")
        
        
        #print("Raw:")
        #print(raw_join_order)
        
        costs = get_costs_for_leftdeep_tree(raw_join_order, card, pred, pred_sel, card_dict)
        
        solution = [raw_join_order, int(costs), (time.time()-start)*1000, False]
        solutions.append(solution)
        if costs < best_costs:
            best_costs = costs
            best_solutions_for_time.append(solution)
            
        # Fallback
        join_order = postprocess_join_order(raw_join_order, cost_vector, num_relations, pred)
        costs = get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, card_dict)
        solution = [join_order, int(costs), (time.time()-start)*1000, True]
        solutions.append(solution)
        if costs < best_costs:
            best_costs = costs
            best_solutions_for_time.append(solution)
        #print("Processed:")
        #print(join_order)
    
    return best_solutions_for_time, solutions


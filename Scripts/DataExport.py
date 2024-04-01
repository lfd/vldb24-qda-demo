#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import prod
import math
import itertools
from math import inf
from sympy.utilities.iterables import multiset_permutations
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.Postprocessing as Postprocessing

import json
import os
import pathlib
import csv
from os import listdir
from os.path import isfile, join
from pathlib import Path

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import time


# In[ ]:


def save_to_csv(data, path, filename):
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()

def load_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with open(datafile, 'rb') as file:
            return json.load(file)
        
def save_data(data, path, filename):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)


# In[ ]:


def is_join_order_valid(join_order, num_card):
    jo_set = set(join_order.copy())
    indices = set(np.arange(num_card))
    
    diff1 = jo_set.difference(indices)
    diff2 = indices.difference(jo_set)
    if len(diff1) != 0 or len(diff2) != 0:
        print("Invalid join order detected.")
        return False
    return True

def export_synthetic_annealing_results(problem_path_prefix, result_path_prefix, output_path, na_cost=20, timeout_in_ms=60000, include_header=True):
    if include_header:
        csv_data = ['method', 'num_relations', 'graph_type', 'problem', 'baseline_cost', 'optimisation_time_in_ms', 'cost', 'normalised_cost']
        save_to_csv(csv_data, output_path, 'synthetic_results_novel.txt')     
    
    graph_types = os.listdir(path=problem_path_prefix + '/synthetic_queries/')
    for graph_type_string in graph_types:
        graph_type = graph_type_string.split("_")[0]
        relations = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph')
        for relations_string in relations:
            i = int(relations_string.split("relations")[0])
            problems = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph/' + str(i) + 'relations')
            for j in problems:
                j = int(j)
                csv_data_list = []
                baseline_cost = inf

                problem_path_main = graph_type + '_graph/' + str(i) + 'relations/' + str(j)
                card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/synthetic_queries/' + problem_path_main, generated_problems=True)
                
                if len(card) < 3 or 0.0 in pred_sel:
                    continue
                
                algorithm_path = result_path_prefix + '/synthetic_queries/' + problem_path_main
                algorithm_result_files = os.listdir(path=algorithm_path)
                
                for algorithm_result_file in algorithm_result_files:
                    algorithm = algorithm_result_file.split(".")[0]
                    algorithm_results = load_data(algorithm_path, algorithm_result_file)
                    if len(algorithm_results) == 0:
                        csv_data_list.append([algorithm, i, graph_type, j, 0, 'n/a', 'n/a', 0])
                        continue
                    
                    # Fetch the last result that is not exceeding the timeout
                    final_algorithm_result = algorithm_results[0]
                    for algorithm_result in algorithm_results:
                        if algorithm_result["time"] <= timeout_in_ms:
                            final_algorithm_result = algorithm_result
                    
                    solution_time = final_algorithm_result["time"]
                    join_order = final_algorithm_result["join_order"]
                    if not is_join_order_valid(join_order, len(card)):
                        return
                    costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                    #if "costs" in final_algorithm_result:
                        #costs = final_algorithm_result["costs"]
                    #elif "cost" in final_algorithm_result:
                        #costs = final_algorithm_result["cost"]
                    
                    if costs < baseline_cost:
                        baseline_cost = costs
                    
                    # In some cases, MILP solution time is slightly above the timeout despite setting the timeout as desired.
                    # We take such cases into consideration by allowing a slight overhead time (1000ms) for MILP
                    if algorithm == 'milp' and solution_time > timeout_in_ms and solution_time <= (timeout_in_ms + 1000):
                        solution_time = timeout_in_ms
                    
                    if solution_time <= timeout_in_ms:
                        csv_data_list.append([algorithm, i, graph_type, j, 0, solution_time, costs, 0])
                    else:
                        csv_data_list.append([algorithm, i, graph_type, j, 0, 'n/a', na_cost, 0])
                
                # Export csv data
                for csv_data in csv_data_list:
                    csv_data[4] = baseline_cost
                    if csv_data[len(csv_data)-2] != 'n/a' and int(baseline_cost) != 0:
                        normalised_cost = csv_data[len(csv_data)-2]/int(baseline_cost)
                        if normalised_cost > na_cost:
                            csv_data[len(csv_data)-1] = na_cost
                        else:
                            csv_data[len(csv_data)-1] = normalised_cost
                    else:
                        csv_data[len(csv_data)-1] = na_cost
                    save_to_csv(csv_data, output_path, 'synthetic_results_novel.txt')
                    
def export_synthetic_annealing_result_times(problem_path_prefix, result_path_prefix, output_path, na_cost=20, timeout_in_ms=60000, include_header=True):
    if include_header:
        csv_data = ['method', 'num_relations', 'graph_type', 'problem', 'baseline_cost', 'optimisation_time_in_ms', 'cost', 'normalised_cost']
        save_to_csv(csv_data, output_path, 'synthetic_result_times.txt')     
    
    graph_types = os.listdir(path=problem_path_prefix + '/synthetic_queries/')
    for graph_type_string in graph_types:
        graph_type = graph_type_string.split("_")[0]
        relations = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph')
        for relations_string in relations:
            i = int(relations_string.split("relations")[0])
            problems = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph/' + str(i) + 'relations')
            for j in problems:
                j = int(j)
                csv_data_list = []
                baseline_cost = inf

                problem_path_main = graph_type + '_graph/' + str(i) + 'relations/' + str(j)
                card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/synthetic_queries/' + problem_path_main, generated_problems=True)
                
                if len(card) < 3 or 0.0 in pred_sel:
                    continue
                
                algorithm_path = result_path_prefix + '/synthetic_queries/' + problem_path_main
                algorithm_result_files = os.listdir(path=algorithm_path)
                
                for algorithm_result_file in algorithm_result_files:
                    algorithm = algorithm_result_file.split(".")[0]
                    algorithm_results = load_data(algorithm_path, algorithm_result_file)
                    if len(algorithm_results) == 0:
                        csv_data_list.append([algorithm, i, graph_type, j, 0, 0, 'n/a', na_cost])
                        csv_data_list.append([algorithm, i, graph_type, j, 0, timeout_in_ms, 'n/a', na_cost])
                        continue
                    
                    csv_data_list.append([algorithm, i, graph_type, j, 0, 0, 'n/a', na_cost])
                    final_costs = na_cost
                    for algorithm_result in algorithm_results:
                        solution_time = algorithm_result["time"]
                        
                        # In some cases, MILP solution time is slightly above the timeout despite setting the timeout as desired.
                        # We take such cases into consideration by allowing a slight overhead time (1000ms) for MILP
                        if algorithm == 'milp' and solution_time > timeout_in_ms and solution_time <= (timeout_in_ms + 1000):
                            solution_time = timeout_in_ms
                        elif solution_time > timeout_in_ms:
                            continue
                            
                        join_order = algorithm_result["join_order"]
                        if not is_join_order_valid(join_order, len(card)):
                            return
                        costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                        final_costs = costs
                        #if "costs" in final_algorithm_result:
                            #costs = final_algorithm_result["costs"]
                        #elif "cost" in final_algorithm_result:
                            #costs = final_algorithm_result["cost"]

                        if costs < baseline_cost:
                            baseline_cost = costs
                        csv_data_list.append([algorithm, i, graph_type, j, 0, solution_time, costs, 0])
                        
                    csv_data_list.append([algorithm, i, graph_type, j, 0, timeout_in_ms, final_costs, 0])
                # Export csv data
                for csv_data in csv_data_list:
                    csv_data[4] = baseline_cost
                    if csv_data[len(csv_data)-2] != 'n/a' and baseline_cost != 0:
                        normalised_cost = csv_data[len(csv_data)-2]/baseline_cost
                        if normalised_cost > na_cost:
                            csv_data[len(csv_data)-1] = na_cost
                        else:
                            csv_data[len(csv_data)-1] = normalised_cost
                    else:
                        csv_data[len(csv_data)-1] = na_cost
                    save_to_csv(csv_data, output_path, 'synthetic_result_times.txt')
                    
def export_benchmark_annealing_results(problem_path_prefix, result_path_prefix, output_path, na_cost=20, timeout_in_ms=60000, include_header=True):
    if include_header:
        csv_data = ['method', 'benchmark', 'problem', 'baseline_cost', 'optimisation_time_in_ms', 'cost', 'normalised_cost']
        save_to_csv(csv_data, output_path, 'benchmark_results_novel.txt')     
    
    query_types = os.listdir(path=problem_path_prefix + '/benchmark_queries/')
    for query_type in query_types:
        queries = os.listdir(path=problem_path_prefix + '/benchmark_queries/' + query_type)
        for query in queries:
            query_number = int(query.split('q')[1])
            csv_data_list = []
            baseline_cost = inf

            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/benchmark_queries/' + query_type + '/' + query, generated_problems=True)
                
            if len(card) < 3 or 0.0 in pred_sel:
                continue
                
            algorithm_path = result_path_prefix + '/benchmark_queries/' + query_type + '/' + query
            algorithm_result_files = os.listdir(path=algorithm_path)
                
            for algorithm_result_file in algorithm_result_files:
                algorithm = algorithm_result_file.split(".")[0]
                if algorithm == 'dpsizelinearCP':
                    continue
                algorithm_results = load_data(algorithm_path, algorithm_result_file)
                if len(algorithm_results) == 0:
                    csv_data_list.append([algorithm, query_type, query, 0, 'n/a', 'n/a', 0])
                    continue
                    
                # Fetch the last result that is not exceeding the timeout
                final_algorithm_result = algorithm_results[0]
                for algorithm_result in algorithm_results:
                    if algorithm_result["time"] <= timeout_in_ms:
                        final_algorithm_result = algorithm_result
                    
                solution_time = final_algorithm_result["time"]
                join_order = final_algorithm_result["join_order"]
                costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                #if "costs" in final_algorithm_result:
                    #costs = final_algorithm_result["costs"]
                #elif "cost" in final_algorithm_result:
                    #costs = final_algorithm_result["cost"]
                    
                if costs < baseline_cost:
                    baseline_cost = costs
                    
                # In some cases, MILP solution time is slightly above the timeout despite setting the timeout as desired.
                # We take such cases into consideration by allowing a slight overhead time (1000ms) for MILP
                if algorithm == 'milp' and solution_time > timeout_in_ms and solution_time <= (timeout_in_ms + 1000):
                    solution_time = timeout_in_ms
                    
                if solution_time <= timeout_in_ms:
                    csv_data_list.append([algorithm, query_type, query, 0, solution_time, costs, 0])
                else:
                    csv_data_list.append([algorithm, query_type, query, 0, 'n/a', na_cost, 0])
                
            # Export csv data
            for csv_data in csv_data_list:
                csv_data[3] = baseline_cost
                if csv_data[len(csv_data)-2] != 'n/a' and baseline_cost != 0:
                    normalised_cost = csv_data[len(csv_data)-2]/baseline_cost
                    if normalised_cost > na_cost:
                        csv_data[len(csv_data)-1] = na_cost
                    else:
                        csv_data[len(csv_data)-1] = normalised_cost
                else:
                    csv_data[len(csv_data)-1] = na_cost
                save_to_csv(csv_data, output_path, 'benchmark_results_novel.txt')


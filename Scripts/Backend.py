#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import prod
import math
import itertools
from math import inf
from sympy.utilities.iterables import multiset_permutations
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.Postprocessing as Postprocessing
import Scripts.DataExport as DataExport

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

from dadk.QUBOSolverDAv2 import QUBOSolverDAv2
from dadk.QUBOSolverCPU import *

import neal

import streamlit as st
from plotnine import *
from plotnine.data import mtcars
import pandas as pd
import networkx as nx
import graphviz as graphviz

import matplotlib
import matplotlib.pyplot

import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dwave.embedding
from dwave.cloud.client import Client

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
        
def load_all_results(path):
    if not os.path.isdir(path):
        return []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for datafile in onlyfiles:
        with open(path + '/' + datafile, 'rb') as file:
            data.append(json.load(file))
    return data

def save_data(data, path, filename):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)
        
def DWave_license_set():
    license_path = 'license'
    if not os.path.exists(license_path + '/dwave.conf' ):
        return False
    return True

# In[ ]:

def get_embedding(bqm, quantum_solver):
    classical_sampler = neal.SimulatedAnnealingSampler()
    QPUGraph = nx.Graph(quantum_solver.edgelist)
    structured_sampler = dimod.StructureComposite(classical_sampler, QPUGraph.nodes, QPUGraph.edges)
    pegasus_embedding_sampler = EmbeddingComposite(structured_sampler)
    
    pegasus_response = pegasus_embedding_sampler.sample(bqm, return_embedding=True)
    pegasus_embedding = pegasus_response.info.get('embedding_context').get('embedding')
    return pegasus_embedding
    
def embed_sampler(sampler, embedding):
    adv_solver = get_adv_solver()
    QPUGraph = nx.Graph(adv_solver.edgelist)
    structured_sampler = dimod.StructureComposite(sampler, QPUGraph.nodes, QPUGraph.edges)
    embedded_sampler = FixedEmbeddingComposite(structured_sampler, embedding=embedding)
    return embedded_sampler
 

def get_adv_solver():
    adv_solver = DWaveSampler(aggregate=False, num_qubits__gt=3000, config_file='../licenses/dwave.conf')
    return adv_solver

def solve_problem_with_quantum_annealing(bqm, sampler, num_reads=20, annealing_time=None):
    sampler = get_adv_solver()
    embedding = get_embedding(bqm, sampler)
    embedded_sampler = embed_sampler(sampler, embedding)
    response = None
    if annealing_time is not None:
        response = embedded_sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time, answer_mode='raw')    
    else:
        response = embedded_sampler.sample(bqm, num_reads=num_reads, answer_mode='raw')
    
    return response


def solve_qubo_with_local_simulated_annealing(qubo, number_runs=100, number_iterations=1000):
    
    sampler = neal.SimulatedAnnealingSampler()
    start = time.time()
    result = sampler.sample(qubo, num_reads=number_runs, num_sweeps=number_iterations, answer_mode='raw')
    opt_time = time.time() - start
    opt_time_in_ms = opt_time * 1000
    
    solutions = []
    for item in result.record:
        bitstring = [int(x) for x in item[0]]
        solutions.append([bitstring, int(item[2]), float(item[1])])
        
    return solutions, opt_time_in_ms

def solve_problem_with_digital_annealing(fujitsu_qubo, da_algorithm='annealing', number_runs=100, number_iterations=1000000, test_with_local_solver=False):
    if test_with_local_solver:
        solver = QUBOSolverCPU(number_runs=number_runs)
    else:
        if da_algorithm == 'annealing':
            solver = QUBOSolverDAv2(optimization_method=da_algorithm, timeout=60, number_iterations=number_iterations, number_runs=number_runs, access_profile_file='annealer.prf', use_access_profile=True)
        else:
            solver = QUBOSolverDAv2(optimization_method=da_algorithm, timeout=60, number_iterations=number_iterations, number_replicas=number_runs, access_profile_file='annealer.prf', use_access_profile=True)

    while True:
        try:
            solution_list = solver.minimize(fujitsu_qubo)
            break
        except Exception as error:
            st.write("Error: ")
            st.write(error)
            #print("Library error. Repeating request")

    execution_time = solution_list.execution_time.total_seconds()
    anneal_time = solution_list.anneal_time.total_seconds()
    solutions = solution_list.solutions
    return solutions, execution_time, anneal_time

def parse_Fujitsu_solutions_for_serialisation(raw_solutions):
    response = []
    for raw_solution in raw_solutions:
        solution = [raw_solution.configuration, float(raw_solution.frequency), float(raw_solution.energy)]
        response.append(solution)
    return response
    
def parse_DWave_solutions_for_serialisation(raw_response):
    response = []
    for i in range(len(raw_response.record)):
        (sample, energy, occ, chain) = raw_response.record[i]
        solution = [sample.tolist(), occ.item(), energy.item()]
        response.append(solution)
    return response

def get_threshold_configuration(approximation_config):
    if approximation_config == 1:
        return (2, [0.63])
    if approximation_config == 2:
        return (2, [2.55])
    if approximation_config == 3:
        return (2, [5.11])
    return (0, [])

def get_DWave_QUBO(card, pred, pred_sel, thres, num_decimal_pos, qubo_penalty_scaling):
    if len(thres) == 0:
        qubo, qubo_matrix_array, penalty_weight = QUBOGenerator.generate_DWave_QUBO_for_left_deep_trees_v2(card, pred, pred_sel, penalty_scaling=qubo_penalty_scaling)
    else:
        qubo, qubo_matrix_array, penalty_weight = QUBOGenerator.generate_DWave_QUBO_for_left_deep_trees(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=qubo_penalty_scaling)
    return qubo, qubo_matrix_array

def get_Fujitsu_QUBO(card, pred, pred_sel, thres, num_decimal_pos, qubo_penalty_scaling):
    if len(thres) == 0:
        qubo, qubo_matrix_array, penalty_weight = QUBOGenerator.generate_Fujitsu_QUBO_for_left_deep_trees_v2(card, pred, pred_sel, penalty_scaling=qubo_penalty_scaling)
    else:
        qubo, qubo_matrix_array, penalty_weight = QUBOGenerator.generate_Fujitsu_QUBO_for_left_deep_trees(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=qubo_penalty_scaling)
    return qubo, qubo_matrix_array

def visualise_QUBO(qubo_matrix_array):
    for i in range(len(qubo_matrix_array)):
        qubo_matrix_array[i][i] = 0
            
    g = nx.from_numpy_matrix(qubo_matrix_array)
    num_nodes = int(len(qubo_matrix_array) * 0.05)
    nodes = np.arange(num_nodes).tolist()
    sg = g.subgraph(nodes)
    fig = matplotlib.pyplot.figure()
    nx.draw(sg, ax=fig.add_subplot(), node_size=20, width=0.2)
    with st.expander("QUBO Graph"):
        st.pyplot(fig)
    
def solve(query_graph_type, num_relations, solver, num_runs, num_annealing_iterations_set, qubo_penalty_scaling_set, approximation_config, visualise_process):
        
    problem_path_main = query_graph_type + '_graph/' + str(num_relations) + 'relations/' + str(0)
    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('Experiments/Problems/synthetic_queries/' + problem_path_main, generated_problems=True)
    baseline_costs = load_data('Experiments/Problems/synthetic_queries/' + problem_path_main, 'baseline_costs.json')[0]
    (num_decimal_pos, thres) = get_threshold_configuration(approximation_config)
    
    qubo = None
    
    dataset = []
    min_result = 20
    max_result = 1
    for num_annealing_iterations_log in range(num_annealing_iterations_set[0], num_annealing_iterations_set[1]+1):
        num_annealing_iterations = pow(10, num_annealing_iterations_log)
        for qubo_penalty_scaling in range(qubo_penalty_scaling_set[0], qubo_penalty_scaling_set[1]+1):
            if solver == 'Local Simulator':
                if qubo == None:
                    st.write('Encode as QUBO...')
                    qubo, qubo_matrix_array = get_DWave_QUBO(card, pred, pred_sel, thres, num_decimal_pos, qubo_penalty_scaling)
                    st.write('Encoding generated!')
                    if visualise_process:
                        visualise_QUBO(qubo_matrix_array)
                st.write('Solve with local simulator...')
                solutions, opt_time_in_ms = solve_qubo_with_local_simulated_annealing(qubo, number_runs=num_runs, number_iterations=num_annealing_iterations)
                st.write('Optimisation finished!')
                st.write('Decoding results...')
                best_solutions_for_time, solutions = Postprocessing.readout([solutions, float(opt_time_in_ms)], card, pred, pred_sel, {}, visualise_process)
                st.write('Results decoded!')
            elif solver == 'Digital Annealer':
                if qubo == None:
                    st.write('Encode as QUBO...')
                    qubo, qubo_matrix_array = get_Fujitsu_QUBO(card, pred, pred_sel, thres, num_decimal_pos, qubo_penalty_scaling)
                    st.write('Encoding generated!')
                    visualise_QUBO(qubo_matrix_array)
                st.write('Solve with Digital Annealer...')
                raw_solutions, execution_time, anneal_time = solve_problem_with_digital_annealing(qubo, da_algorithm='annealing', number_runs=num_runs, number_iterations=num_annealing_iterations, test_with_local_solver=False)
                st.write('Optimisation finished!')
                st.write('Decoding results...')
                solutions = parse_Fujitsu_solutions_for_serialisation(raw_solutions)
                [solutions, float(execution_time), float(anneal_time)]
                best_solutions_for_time, solutions = Postprocessing.readout([solutions, float(execution_time), float(anneal_time)], card, pred, pred_sel, {}, visualise_process)
                st.write('Results decoded!')
            elif solver == 'Quantum Processing Unit':
                if not DWave_license_set():
                    st.write('Error: D-Wave License File Missing!')
                    return
                
                if qubo == None:
                    st.write('Encode as QUBO...')
                    qubo, qubo_matrix_array = get_DWave_QUBO(card, pred, pred_sel, thres, num_decimal_pos, qubo_penalty_scaling)
                    st.write('Encoding generated!')
                    visualise_QUBO(qubo_matrix_array)
                st.write('Solve with Quantum Annealer...')
                raw_solutions = solve_problem_with_quantum_annealing(bqm, sampler, num_reads=20, annealing_time=None)
                st.write('Optimisation finished!')
                st.write('Decoding results...')
                solutions = parse_DWave_solutions_for_serialisation(raw_solutions)
                best_solutions_for_time, solutions = Postprocessing.readout([solutions, float(execution_time), float(anneal_time)], card, pred, pred_sel, {}, visualise_process)
                st.write('Results decoded!')
            else:
                st.write('Unsupported solver. Abort')
                return
    
            for i in range(len(solutions)):
                solution = solutions[i]
                jo_costs = solution[1] / baseline_costs
                if jo_costs > 20:
                    jo_costs = 20
                if jo_costs < min_result:
                    min_result = jo_costs
                if jo_costs > max_result:
                    max_result = jo_costs
                dataset.append([i, str(num_annealing_iterations_log), str(qubo_penalty_scaling), jo_costs])
    
    st.write('Plotting results...')
    df = pd.DataFrame(dataset, columns=['Shot', '#Annealing Iterations (log)', 'Penalty Scaling', 'Normalised Costs'])    
        
    p = ggplot(df, aes(x="#Annealing Iterations (log)", y="Normalised Costs")) + geom_boxplot() + geom_point(df, aes(y="Normalised Costs")) + scale_x_discrete() + scale_y_log10(breaks = [1.0, 5.0, 10.0, 20.0], limits=[1.0,20.0], labels=[1,5, 10,"N/A"])
    st.pyplot(ggplot.draw(p))
    
    p = ggplot(df, aes(x="Penalty Scaling", y="Normalised Costs")) + geom_boxplot() + geom_point(df, aes(y="Normalised Costs")) + scale_x_discrete() + scale_y_log10(breaks = [1.0, 5.0, 10.0, 20.0], limits=[1.0,20.0], labels=[1,5, 10,"N/A"])
    st.pyplot(ggplot.draw(p))


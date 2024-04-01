'''
Created on Mar 23, 2024

@author: immanueltrummer
'''
import streamlit as st
import Scripts.Backend as Backend


def solve(
    query_graph_type, num_relations, solver, 
    num_runs, num_annealing_iterations, qubo_penalty_scaling, 
    approximation_config, visualise_process):
    st.write('Starting annealing run!')
    Backend.solve(query_graph_type, num_relations, solver, num_runs, num_annealing_iterations, qubo_penalty_scaling, approximation_config, visualise_process)

if __name__ == '__main__':
    
    st.set_page_config(page_title='Digital and Quantum Annealing Demo')
    st.header('Demonstrating Quantum(-Inspired) Computing')
    
    with st.expander('Benchmark'):
        query_graph_type = st.selectbox(
            label='Query Graph Type', 
            options=['Chain', 'Star', 'Cycle'])
        num_relations = st.slider(
            label='Number of Relations', 
            min_value=18, max_value=50, step=4)
    
    with st.expander('Solver'):
        solver = st.selectbox(
            label='Solver Type', 
            options=[
                'Quantum Processing Unit', 
                'Digital Annealer', 
                'Local Simulator'])
        num_runs = st.slider(
            label='Number of Optimisation Runs', 
            min_value=20, max_value=100, step=10)
        num_annealing_iterations = st.slider(
            label='Number of Iterations (log)', 
            min_value=2, max_value=6, value=(2, 4))
        qubo_penalty_scaling = st.slider(
            label='Qubo Penalty Scaling', 
            min_value=1, max_value=10, value=(1, 5))
        approximation_config = st.selectbox(
            label='Approximation Configuration', 
            options=list(range(1,4)))
        visualise_process = st.checkbox(label='Visualise En-/Decoding', value=True)
    
    if st.button('Start'):
        solve(
            query_graph_type, num_relations, solver, 
            num_runs, num_annealing_iterations, qubo_penalty_scaling, 
            approximation_config, visualise_process)
# Quantum-Inspired Digital Annealing for Join Ordering

This repository contains code for "Demonstrating Quantum(-Inspired) Computing for Join Ordering", submitted to VLDB 2024.

## Project Structure

Code for the UI is contained in gui.py, and user inputs are processed in Scripts/backend.py depending on the selected solver. Scripts/QUBOGenerator.py constructs JO-QUBO encodings based on our formulation method, allowing their deployment on quantum(-inspired) HW. Finally, Scripts/Postprocessing.py contains code for reading out join orders from each annealing solution obtained by the annealing device. 

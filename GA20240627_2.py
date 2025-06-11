import random
import struct
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import psutil
import os

population_size = 50  
crossover_probability = 0.8  
mutation_probability = 0.05  
elitism = True  
max_generations = 40 


def read_data_from_file(file_path, start_row, end_row):
    df = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=end_row - start_row, low_memory=False)
    third_column = df.iloc[:, 2].tolist() 
    return third_column


def convert_to_binary(third_column):
    binary_values = []
    for data in third_column:
        packed_double = struct.pack('>d', float(data))  
        binary_value = ''.join(f'{byte:08b}' for byte in packed_double) 
        binary_values.append(binary_value)
    return binary_values


def xor_data(binary_values):
    xor_results = []
    prev_value = None
    for value in binary_values:
        if prev_value is not None:
            xor_result = int(value, 2) ^ int(prev_value, 2) 
            xor_results.append(format(xor_result, '064b'))  
        prev_value = value
    return xor_results


def count_zeros_until_one(xor_results):
    count_zeros = 0
    found_one = False
    for bit in xor_results:
        if bit == '0' and not found_one:
            count_zeros += 1
        elif bit == '1':
            break
    return count_zeros

def fitness_function(genome, zero_indices=None):
    if zero_indices is None:
        zero_indices = np.where(genome == 0)[0]

    min_position = 4
    max_position = 36
    max_step = 4
    satisfied_conditions = 0

    
    for index in zero_indices:
        if min_position <= index <= max_position:
            satisfied_conditions += 1

   
    for i in range(1, len(zero_indices)):
        step_size = zero_indices[i] - zero_indices[i - 1]
        if 1 < step_size <= max_step:
            satisfied_conditions += 1

    if satisfied_conditions == 0:
        total_fitness = float('inf')  
    else:
        total_fitness = 1 / satisfied_conditions  

    return total_fitness



def initialize_population(populationSize, xor_results):
    encodelength = 63  
    L_count = []
    for data in xor_results:
        zeros_count = count_zeros_until_one(data)
        L_count.append(zeros_count)
    chromosomes = np.ones((populationSize, encodelength), dtype=np.uint8)
    unique_L_count = list(set(L_count))
    for i in range(populationSize):
        selected_counts = random.sample(unique_L_count, 7)
        chromosome = np.ones(encodelength, dtype=np.uint8)
        for count in selected_counts:
            if count < encodelength: 
                chromosome[count] = 0
        chromosomes[i] = chromosome

    return chromosomes



def selection(population, fitness_values, tournament_size=3):
    parents = []
    for _ in range(len(population)):
        if len(population) <= 3:
            tournament_indices = random.sample(range(len(population)), len(population))
        else:
            tournament_indices = random.sample(range(len(population)), tournament_size)
        best_index = min(tournament_indices, key=lambda idx: fitness_values[idx])
        selected_individual = population[best_index]
        parents.append(selected_individual)
    return parents


def crossover(parents, crossover_probability):
    children = []
    for _ in range(len(parents) // 2):
        if random.random() < crossover_probability:
            parent1, parent2 = random.sample(parents, 2)
            crossover_points = sorted(random.sample(range(1, len(parent1)), 3))
            child1 = parent1.copy()  
            child2 = parent2.copy()  
            for i, point in enumerate(crossover_points + [len(parent1)]):
                if i % 2 == 0:
                    child1[:point], child2[:point] = parent1[:point], parent2[:point]
                else:
                    child1[:point], child2[:point] = parent2[:point], parent1[:point]
            children.extend([child1, child2])
        else:
            children.extend(random.sample(parents, 2))
    return children


def mutation(children, mutation_probability):
    for i in range(len(children)):
        if random.random() < mutation_probability:
            zero_indices = np.where(children[i] == 0)[0]
            one_indices = np.where(children[i] == 1)[0]
            if len(zero_indices) > 0 and len(one_indices) > 0:
                zero_index = random.choice(zero_indices)
                one_index = random.choice(one_indices)
                children[i][zero_index] = 1
                children[i][one_index] = 0
    return children


def run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations, xor_results):
    global zero_positions
    start_time = time.time()  
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 
    population = initialize_population(population_size, xor_results)
    best_genome = None
    time_taken = []
    memory_usage = []
    best_fitness_values = [] 
    best_genomes = [] 

    for generation in range(max_generations):
        generation_start_time = time.time()

        fitness_values = [fitness_function(genome) for genome in population]
        best_fitness_value = min(fitness_values)
        best_genome = population[fitness_values.index(best_fitness_value)]

        
        best_fitness_values.append(best_fitness_value)
        best_genomes.append(best_genome)

        parents = selection(population, fitness_values)
        children = crossover(parents, crossover_probability)
        children = mutation(children, mutation_probability)

        population = children

       
        generation_end_time = time.time()
        generation_time =  generation_start_time
        time_taken.append(generation_time)

        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  
        memory_usage.append(current_memory - start_memory)
        zero_positions = np.where(best_genome == 0)[0].tolist()

        print(f"Generation: {generation}, Best fitness: {best_fitness_value}, Best genome: {best_genome}, Time: {generation_time:.4f}s")

    end_time = time.time()  
    total_time = end_time - start_time  
    print(f"Total running time: {total_time:.4f} seconds") 
    print("\nPositions of zeros in the best genome:", zero_positions)

    # plt.rcParams.update({
    #     'figure.dpi': 300, 
    #     # 'axes.titles.fontsize': 10,  
    #     'axes.labelsize': 14, 
    #     'xtick.labelsize': 12,  
    #     'ytick.labelsize': 12,  
    #     'legend.fontsize': 12  
    # })

    
    # plt.figure(figsize=(12, 6))
    #
    
    # plt.subplot(1, 2, 1)
    # plt.plot(range(max_generations), time_taken, label="Time overhead (s)")
    # plt.xlabel("Generation")
    # plt.ylabel("Time (seconds)")
    # plt.title("Time Complexity Analysis")
    # plt.legend()
    #
   
    # plt.subplot(1, 2, 2)
    # plt.plot(range(max_generations), memory_usage, label="Memory usage (MB)")
    # plt.xlabel("Generation")
    # plt.ylabel("Memory Usage (MB)")
    # plt.title("Space Complexity Analysis")
    # plt.legend()
    #
   
    # plt.tight_layout()
    # plt.show()
    #
   
    # plt.figure(figsize=(6, 6))
    # plt.plot(range(max_generations), best_fitness_values, label="Best Fitness per Generation")
    # plt.xlabel("Generation")
    # plt.ylabel("Best Fitness")
    # plt.title("Convergence of Fitness")
    # plt.legend()
    #
   
    # plt.tight_layout()
    # plt.show()

    return best_genome


best_genome = run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations, xor_results)
print(f"Solution found: {best_genome}")

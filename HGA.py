import random
from collections import Counter
import numpy as np
from scipy.optimize import fsolve, basinhopping
import random
import time
import timeit
import gzip
import pandas as pd
import struct
population_size = 100
crossover_probability = 0.8
mutation_probability = 0.01
elitism = True
max_generations = 100



df = pd.read_csv('E:\chimp-main\src\IoT1.csv', nrows=20000)


third_column = df.iloc[:, 2].tolist() 


# print(third_column)

# print(type(third_column))


binary_values = []
for data in third_column:
    packed_double = struct.pack('>d', float(data))  
    binary_value = ''.join(f'{byte:08b}' for byte in packed_double)  
    binary_values.append(binary_value)
# print(len(third_column))


# for value in binary_values:
#     print(value)

xor_results = []
prev_value = None
for value in binary_values:
    if prev_value is not None:
        xor_result = int(value, 2) ^ int(prev_value, 2)
        xor_results.append(format(xor_result, '064b')) 
    prev_value = value
# print(xor_results)
# for result in xor_results:
#     print(result)

def count_zeros_until_one(xor_results):
    count_zeros = 0
    found_one = False

    for bit in xor_results:
        if bit == '0' and not found_one:
            count_zeros += 1
        elif bit == '1':
            break

    return count_zeros

def fitness_function(genome):
    min_position = 8
    max_position = 32
    max_step = 4

    zero_indices = np.where(genome == 0)[0]

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


#########################################################
def initialize_population(populationSize):
    encodelength = 63 
    # encodelength = 31
    # temp = []
    L_count = []
    for data in xor_results:
        zeros_count = count_zeros_until_one(data)
        L_count.append(zeros_count)
    chromosomes = np.ones((populationSize, encodelength), dtype=np.uint8)
    unique_L_count = list(set(L_count))
    for i in range(populationSize):
        # temp = L_count
        selected_counts = random.sample(unique_L_count, 7)
        chromosome = np.ones(encodelength, dtype=np.uint8)
        for count in selected_counts:
            if count < encodelength: 
                chromosome[count] = 0
                # chromosome[0] = 0
        chromosomes[i] = chromosome

    print(chromosomes)
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaa',len(chromosomes))
    return chromosomes

def selection(population, fitness_values, tournament_size=3):
        parents = []
        # print('0000000000000000000000',len(population))
        for _ in range(len(population)):
            if len(population) <= 3:
                tournament_indices = random.sample(range(len(population)), len(population))
                # continue
            else:
                tournament_indices = random.sample(range(len(population)), tournament_size
            best_index = min(tournament_indices, key=lambda idx: fitness_values[idx])
            selected_individual = population[best_index]
            print('individuals', selected_individual)
            print('index', best_index)
            parents.append(selected_individual)
            # print('parents',parents)
        return parents

def crossover(parents, crossover_probability):
    children = []
    for _ in range(len(parents) // 2):
        if random.random() < crossover_probability:
            parent1, parent2 = random.sample(parents, 2)
            crossover_points = sorted(random.sample(range(1, len(parent1)), 3))
            child1, child2 = parent1.copy(), parent2.copy()
            for i, point in enumerate(crossover_points + [len(parent1)]):
                if i % 2 == 0:
                    child1[:point], child2[:point] = parent1[:point], parent2[:point]
                else:
                    child1[:point], child2[:point] = parent2[:point], parent1[:point]
            child1, child2 = correct_zeros(child1, child2, 7, 7)
            children.extend([child1, child2])
        else:
            children.extend(random.sample(parents, 2))
    return children

def correct_zeros(child1, child2, num_zeros1, num_zeros2):
    indices1 = np.where(child1 == 1)[0]
    indices2 = np.where(child2 == 1)[0]

    diff1 = num_zeros1 - np.sum(child1 == 0)
    diff2 = num_zeros2 - np.sum(child2 == 0)

    for _ in range(abs(diff1)):
        if diff1 > 0:
            idx = random.choice(indices1)
            child1[idx] = 0
        elif diff1 < 0:
            idx = random.choice(np.where(child1 == 0)[0])
            child1[idx] = 1

    for _ in range(abs(diff2)):
        if diff2 > 0:
            idx = random.choice(indices2)
            child2[idx] = 0
        elif diff2 < 0:
            idx = random.choice(np.where(child2 == 0)[0])
            child2[idx] = 1

    return child1, child2

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


def run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations):
    start_time = time.time()  

    population = initialize_population(population_size) 
    best_genome = None

    for generation in range(max_generations): 
        fitness_values = []
        for genome in population:
            fitness_values.append(fitness_function(genome))

        best_fitness_value = min(fitness_values) 
        print('best_fitness_value', best_fitness_value)
        best_genome = population[fitness_values.index(best_fitness_value)]

        parents = selection(population, fitness_values) 
        print('parents', parents)
        children = crossover(parents, crossover_probability)
        children = mutation(children, mutation_probability) 

        population = children

        print(f"Generation: {generation}, Best fitness: {best_fitness_value}, Best genome: {best_genome}")

    end_time = time.time()  
    total_time = end_time - start_time  
    print(f"Total running time: {total_time:.4f} seconds") 

    return best_genome


best_genome = run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations)
print(f"Solution found: {best_genome}")


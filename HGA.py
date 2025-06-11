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
# 遗传算法参数
population_size = 100#种群数
crossover_probability = 0.8#交叉概率//这两个概率是否可以更改
mutation_probability = 0.01#变异概率
elitism = True#精英主义
max_generations = 100#最大代数

# 思路：
# 1、导入数据集（完成）
# 2、将数据进行异或操作后，得到所有前导零的个数（完成）
# 3、根据前导零个数进行最优值挑选

# 读取CSV文件
df = pd.read_csv('E:\chimp-main\src\IoT1.csv', nrows=20000)

# 提取第三列的数据
third_column = df.iloc[:, 2].tolist()  # iloc[:, 2]表示第三列，.tolist()将其转换为列表

# # 打印第三列的数据
# print(third_column)
# #查看类型
# print(type(third_column))
# 将每条数据转换为IEEE 754标准下的64位双精度二进制表示形式

binary_values = []
for data in third_column:
    packed_double = struct.pack('>d', float(data))  # '>d' 表示以大端序打包为双精度浮点数
    binary_value = ''.join(f'{byte:08b}' for byte in packed_double)  # 将每个字节转换为二进制字符串并连接
    binary_values.append(binary_value)
# print(len(third_column))

# # 打印转换后的结果
# for value in binary_values:
#     print(value)

# 对每条数据与其上一条数据进行异或操作
xor_results = []
prev_value = None
for value in binary_values:
    if prev_value is not None:
        xor_result = int(value, 2) ^ int(prev_value, 2)  # 将二进制字符串转换为整数后进行异或操作
        xor_results.append(format(xor_result, '064b'))  # 将异或结果转换为64位二进制字符串并存储
    prev_value = value
# print(xor_results)
# # 打印异或操作的结果
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
    """
     计算给定基因组的适应度值。

     条件：
     1. 每个'0'之间需要存在一定的步长，但最大步长不能超过max_step。
     2. 每个'0'之间的步长不能为0。
     3. '0'的位置最低不能低于min_position，最大不能超过max_position。
     4. 每个'0'不能连续出现。
     5. 适应度值越大越好。

     参数:
     genome: numpy数组，表示基因组，其中值可以是0或1。
     min_position: '0'的位置最低值。
     max_position: '0'的位置最大值。
     max_step: 两个'0'之间的最大步长。

     返回:
     total_fitness: 总的适应度值，为满足条件的数量之和的倒数。
     """

    # 找到基因组中所有值为0的位置索引
    zero_indices = np.where(genome == 0)[0]

    # 计算满足条件的数量
    satisfied_conditions = 0

    # 确保0的位置在[min_position, max_position]之间
    for index in zero_indices:
        if min_position <= index <= max_position:
            satisfied_conditions += 1

    # 确保每对连续的0之间的步长不超过max_step
    for i in range(1, len(zero_indices)):
        step_size = zero_indices[i] - zero_indices[i - 1]
        if 1 < step_size <= max_step:
            satisfied_conditions += 1

    # 避免除以零
    if satisfied_conditions == 0:
        total_fitness = float('inf')  # 无效解的惩罚
    else:
        total_fitness = 1 / satisfied_conditions  # 满足条件的数量之和的倒数

    return total_fitness


#########################################################
# 初始化种群
def initialize_population(populationSize):
    encodelength = 63  # 示例编码长度
    # encodelength = 31
    # temp = []
    L_count = []
    for data in xor_results:
        zeros_count = count_zeros_until_one(data)
        L_count.append(zeros_count)
    # 初始化 chromosomes 数组
    chromosomes = np.ones((populationSize, encodelength), dtype=np.uint8)
    unique_L_count = list(set(L_count))
    #为每个个体随机挑选 7 个不同的前导零个数值，并设置这些位置的值为 '0'
    for i in range(populationSize):
        # temp = L_count
        selected_counts = random.sample(unique_L_count, 7)
        chromosome = np.ones(encodelength, dtype=np.uint8)
        for count in selected_counts:
            if count < encodelength:  # 确保前导零个数值不超出编码长度
                chromosome[count] = 0
                # chromosome[0] = 0
        chromosomes[i] = chromosome

    print(chromosomes)
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaa',len(chromosomes))
    return chromosomes

def selection(population, fitness_values, tournament_size=3):
        # 锦标赛选择
        parents = []
        # print('0000000000000000000000',len(population))
        for _ in range(len(population)):
            # 从种群中随机选择 tournament_size 个个体
            if len(population) <= 3:
                tournament_indices = random.sample(range(len(population)), len(population))
                # continue
            else:
                tournament_indices = random.sample(range(len(population)), tournament_size)
            # 在锦标赛中选择适应度最小的个体
            best_index = min(tournament_indices, key=lambda idx: fitness_values[idx])
            # 获取选择的个体
            selected_individual = population[best_index]
            print('individuals', selected_individual)
            print('index', best_index)
            # 将选择的个体添加到父母列表中
            parents.append(selected_individual)
            # print('parents',parents)
        return parents

def crossover(parents, crossover_probability):
    children = []
    for _ in range(len(parents) // 2):
        if random.random() < crossover_probability:
            parent1, parent2 = random.sample(parents, 2)
            # 多点交叉的具体实现，例如：
            crossover_points = sorted(random.sample(range(1, len(parent1)), 3))
            child1, child2 = parent1.copy(), parent2.copy()
            for i, point in enumerate(crossover_points + [len(parent1)]):
                if i % 2 == 0:
                    child1[:point], child2[:point] = parent1[:point], parent2[:point]
                else:
                    child1[:point], child2[:point] = parent2[:point], parent1[:point]
            # 调整子代个体中的 '0' 数量
            child1, child2 = correct_zeros(child1, child2, 7, 7)
            children.extend([child1, child2])
        else:
            children.extend(random.sample(parents, 2))
    return children

def correct_zeros(child1, child2, num_zeros1, num_zeros2):
    # 获取子代个体中 '1' 的索引位置
    indices1 = np.where(child1 == 1)[0]
    indices2 = np.where(child2 == 1)[0]

    # 计算实际和预期的零差值
    diff1 = num_zeros1 - np.sum(child1 == 0)
    diff2 = num_zeros2 - np.sum(child2 == 0)

    # 交换位置以匹配预期的零数量
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

# 变异
def mutation(children, mutation_probability):
    for i in range(len(children)):
        if random.random() < mutation_probability:
            # 找到所有 0 和 1 的索引
            zero_indices = np.where(children[i] == 0)[0]
            one_indices = np.where(children[i] == 1)[0]

            if len(zero_indices) > 0 and len(one_indices) > 0:
                # 随机选择一个 0 和一个 1 的索引进行交换
                zero_index = random.choice(zero_indices)
                one_index = random.choice(one_indices)

                # 进行变异：0 变成 1，1 变成 0
                children[i][zero_index] = 1
                children[i][one_index] = 0

    return children

# 运行遗传算法
def run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations):
    start_time = time.time()  # 记录开始时间

    population = initialize_population(population_size)  # 得到10个64位的随机数
    best_genome = None

    for generation in range(max_generations):  # 最大迭代次数
        fitness_values = []
        for genome in population:
            fitness_values.append(fitness_function(genome))

        best_fitness_value = min(fitness_values)  # 取最大适应度
        print('best_fitness_value', best_fitness_value)
        best_genome = population[fitness_values.index(best_fitness_value)]  # 最大适应度对应的个体

        parents = selection(population, fitness_values)  # 按比例选择
        print('parents', parents)
        children = crossover(parents, crossover_probability)
        children = mutation(children, mutation_probability)  # 先不考虑变异

        population = children

        print(f"Generation: {generation}, Best fitness: {best_fitness_value}, Best genome: {best_genome}")

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算运行时间
    print(f"Total running time: {total_time:.4f} seconds")  # 输出运行时间

    return best_genome


# 运行算法
best_genome = run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations)
print(f"Solution found: {best_genome}")


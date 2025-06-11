import random
import struct
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import psutil
import os

# 遗传算法参数
population_size = 50  # 种群数
crossover_probability = 0.8  # 交叉概率
mutation_probability = 0.05  # 变异概率
elitism = True  # 精英主义
max_generations = 40  # 最大代数

# 读取文件的某一段数据
def read_data_from_file(file_path, start_row, end_row):
    df = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=end_row - start_row, low_memory=False)
    third_column = df.iloc[:, 2].tolist()  # 提取第三列的数据
    return third_column

# 将数据转换为IEEE 754标准下的64位双精度二进制表示形式
def convert_to_binary(third_column):
    binary_values = []
    for data in third_column:
        packed_double = struct.pack('>d', float(data))  # '>d' 表示以大端序打包为双精度浮点数
        binary_value = ''.join(f'{byte:08b}' for byte in packed_double)  # 将每个字节转换为二进制字符串并连接
        binary_values.append(binary_value)
    return binary_values

# 对每条数据与其上一条数据进行异或操作
def xor_data(binary_values):
    xor_results = []
    prev_value = None
    for value in binary_values:
        if prev_value is not None:
            xor_result = int(value, 2) ^ int(prev_value, 2)  # 将二进制字符串转换为整数后进行异或操作
            xor_results.append(format(xor_result, '064b'))  # 将异或结果转换为64位二进制字符串并存储
        prev_value = value
    return xor_results

# 计算前导零的数量
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

    # 确保0的位置在[min_position, max_position]之间
    for index in zero_indices:
        if min_position <= index <= max_position:
            satisfied_conditions += 1

    # 确保每对连续的0之间的步长不超过max_step
    for i in range(1, len(zero_indices)):
        step_size = zero_indices[i] - zero_indices[i - 1]
        if 1 < step_size <= max_step:
            satisfied_conditions += 1

    if satisfied_conditions == 0:
        total_fitness = float('inf')  # 无效解的惩罚
    else:
        total_fitness = 1 / satisfied_conditions  # 满足条件的数量之和的倒数

    return total_fitness


#初始化种群
def initialize_population(populationSize, xor_results):
    encodelength = 63  # 示例编码长度
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
            if count < encodelength:  # 确保前导零个数值不超出编码长度
                chromosome[count] = 0
        chromosomes[i] = chromosome

    return chromosomes


# 选择操作（锦标赛选择）
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
            child1 = parent1.copy()  # 只创建一个副本
            child2 = parent2.copy()  # 只创建一个副本
            for i, point in enumerate(crossover_points + [len(parent1)]):
                if i % 2 == 0:
                    child1[:point], child2[:point] = parent1[:point], parent2[:point]
                else:
                    child1[:point], child2[:point] = parent2[:point], parent1[:point]
            children.extend([child1, child2])
        else:
            children.extend(random.sample(parents, 2))
    return children

# 变异操作
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

# 运行遗传算法并分析时间和空间复杂度
def run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations, xor_results):
    global zero_positions
    start_time = time.time()  # 记录开始时间
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 记录初始内存占用 (MB)
    population = initialize_population(population_size, xor_results)
    best_genome = None
    time_taken = []
    memory_usage = []
    best_fitness_values = []  # 新增：记录每代最优适应度
    best_genomes = []  # 新增：记录每代最优基因组

    for generation in range(max_generations):
        generation_start_time = time.time()

        fitness_values = [fitness_function(genome) for genome in population]
        best_fitness_value = min(fitness_values)
        best_genome = population[fitness_values.index(best_fitness_value)]

        # 记录每代最优适应度和基因组
        best_fitness_values.append(best_fitness_value)
        best_genomes.append(best_genome)

        parents = selection(population, fitness_values)
        children = crossover(parents, crossover_probability)
        children = mutation(children, mutation_probability)

        population = children

        # 记录时间和内存使用情况
        generation_end_time = time.time()
        generation_time =  generation_start_time
        time_taken.append(generation_time)

        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 当前内存占用 (MB)
        memory_usage.append(current_memory - start_memory)
        zero_positions = np.where(best_genome == 0)[0].tolist()

        print(f"Generation: {generation}, Best fitness: {best_fitness_value}, Best genome: {best_genome}, Time: {generation_time:.4f}s")

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算运行时间
    print(f"Total running time: {total_time:.4f} seconds")  # 输出运行时间
    print("\nPositions of zeros in the best genome:", zero_positions)

    # plt.rcParams.update({
    #     'figure.dpi': 300,  # 设置图像的 DPI 为 300，提高分辨率
    #     # 'axes.titles.fontsize': 10,  # 设置标题的字体大小
    #     'axes.labelsize': 14,  # 设置坐标轴标签的字体大小
    #     'xtick.labelsize': 12,  # 设置 x 轴刻度标签的字体大小
    #     'ytick.labelsize': 12,  # 设置 y 轴刻度标签的字体大小
    #     'legend.fontsize': 12  # 设置图例字体大小
    # })

    # # 绘制适应度随代数的变化
    # plt.figure(figsize=(12, 6))
    #
    # # 时间复杂度图
    # plt.subplot(1, 2, 1)
    # plt.plot(range(max_generations), time_taken, label="Time overhead (s)")
    # plt.xlabel("Generation")
    # plt.ylabel("Time (seconds)")
    # plt.title("Time Complexity Analysis")
    # plt.legend()
    #
    # # 空间复杂度图
    # plt.subplot(1, 2, 2)
    # plt.plot(range(max_generations), memory_usage, label="Memory usage (MB)")
    # plt.xlabel("Generation")
    # plt.ylabel("Memory Usage (MB)")
    # plt.title("Space Complexity Analysis")
    # plt.legend()
    #
    # # 自动调整子图布局并显示
    # plt.tight_layout()
    # plt.show()
    #
    # # 适应度收敛图
    # plt.figure(figsize=(6, 6))
    # plt.plot(range(max_generations), best_fitness_values, label="Best Fitness per Generation")
    # plt.xlabel("Generation")
    # plt.ylabel("Best Fitness")
    # plt.title("Convergence of Fitness")
    # plt.legend()
    #
    # # 显示图形并去除多余的空白
    # plt.tight_layout()
    # plt.show()

    return best_genome

# 读取文件中的一段数据
file_path = 'E:/chimp-main/src/Lightning.csv'
start_row =0  # 设置开始行
end_row = 12300  # 设置结束行

third_column = read_data_from_file(file_path, start_row, end_row)
binary_values = convert_to_binary(third_column)
xor_results = xor_data(binary_values)

# 运行遗传算法
best_genome = run_genetic_algorithm(population_size, crossover_probability, mutation_probability, max_generations, xor_results)
print(f"Solution found: {best_genome}")

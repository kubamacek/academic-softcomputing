import ctypes
import itertools
import math
import random
import struct
import sys

# RANGE_START = 2.7
# RANGE_END = 7.5
RANGE_START = 0
RANGE_END = 1.2

NUMBER_OF_GENERATIONS = 30
INITIAL_SIZE_OF_POPULATION = 30
MAX_SIZE_OF_POPULATION = 500

BEST_PARENTS_CHOSEN = 10
BASE_PERCENT_CHANCE_TO_BECOME_PARENT = 20

MUTATION_PERCENT_CHANCE = 10
SINGLE_BYTE_MUTATION_CHANCE = 15


def calculate_function(x):
    return -(1.4 - 3 * x) * math.sin(18 * x)


# def calculate_function(x):
#     return math.sin(x) + math.sin((10 / 3) * x)


def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def initialization():
    x_population = set()
    for i in range(0, INITIAL_SIZE_OF_POPULATION):
        x_population.add(random.uniform(RANGE_START, RANGE_END))
    return x_population


def evaluation(x_population):
    x_population_with_results = dict()
    for x in x_population:
        x_population_with_results[x] = calculate_function(x)

    x_population_with_results = dict(sorted(x_population_with_results.items(), key=lambda item: item[1]))
    return x_population_with_results


def calculate_probability(chance_in_percents):
    return random.randint(0, 100) <= chance_in_percents


def selection(x_population_with_results):
    x_population_sorted = list(x_population_with_results.keys())
    parents = x_population_sorted[:BEST_PARENTS_CHOSEN]

    # chooses parents depending by randomness and chance. Can be skipped
    population_size = len(x_population_sorted)
    for index in range(BEST_PARENTS_CHOSEN, population_size):
        percent_chance = (BASE_PERCENT_CHANCE_TO_BECOME_PARENT / population_size) * (population_size - index)
        if calculate_probability(percent_chance):
            parents.append(x_population_sorted[index])

    return parents


def combine_parents(parent1, parent2):
    parent1_bin = float_to_bin(parent1)
    parent2_bin = float_to_bin(parent2)
    child_element = ''
    if len(parent1_bin) != len(parent2_bin):
        raise Exception

    for index in range(0, len(parent1_bin)):
        if parent1_bin[index] == parent2_bin[index]:
            child_element += parent1_bin[index]
        else:
            if calculate_probability(50):
                child_element += parent1_bin[index]
            else:
                child_element += parent2_bin[index]
    return bin_to_float(child_element)


def crossover(parents):
    offspring = list()
    parents_combinations = itertools.combinations(parents, 2)
    for parents in parents_combinations:
        child_element = combine_parents(parents[0], parents[1])
        while not RANGE_START <= child_element <= RANGE_END:
            child_element = combine_parents(parents[0], parents[1])
        offspring.append(child_element)
    return offspring


def perform_mutation(element):
    element_bin = float_to_bin(element)
    mutated_element_bin = ''
    for index in range(0, len(element_bin)):
        if calculate_probability(SINGLE_BYTE_MUTATION_CHANCE):
            if element_bin[index] == '0':
                mutated_element_bin += '1'
            if element_bin[index] == '1':
                mutated_element_bin += '0'
        else:
            mutated_element_bin += element_bin[index]

    return bin_to_float(mutated_element_bin)


def mutation(offspring):
    for element in offspring:
        if calculate_probability(MUTATION_PERCENT_CHANCE):
            mutated_element = perform_mutation(element)
            while not RANGE_START <= mutated_element <= RANGE_END:
                mutated_element = perform_mutation(element)
            offspring.append(mutated_element)

    return offspring


def print_results(x_population):
    counter = 0
    results = evaluation(x_population)
    for element in results.keys():
        counter += 1
        print("{}. x:{} value:{}".format(counter, element, results[element]))
        if counter == 5:
            exit()


def cleanup_population(x_population):
    if len(x_population) <= 30:
        return x_population

    if INITIAL_SIZE_OF_POPULATION < int(len(x_population) / 2) < MAX_SIZE_OF_POPULATION:
        return x_population[:int(len(x_population) / 2)]

    return x_population[:MAX_SIZE_OF_POPULATION]


def genetic_algorithm():
    x_population = initialization()
    for i in range(0, NUMBER_OF_GENERATIONS):
        print("generation: ", i, "generation size: ", len(x_population))
        x_population_with_results = evaluation(x_population)
        x_population = set(cleanup_population(list(x_population_with_results.keys())))
        x_population_with_results = evaluation(x_population)
        parents = selection(x_population_with_results)
        offspring = crossover(parents)
        x_population.update(mutation(offspring))

    print_results(x_population)


genetic_algorithm()

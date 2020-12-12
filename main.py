"""
This program performs Evolution Strategy Algorithm for finding egg holder function global minimum
"""
from math import sqrt, exp, sin
from random import randrange, randint, uniform, normalvariate
from typing import List


class Chromosome:
    """
    class for chromosomes.
    """
    x: float
    y: float
    sigma: float

    def __init__(self, x_in, y_in, sigma_in):
        self.x = x_in
        self.y = y_in
        self.sigma = sigma_in

    def mutate(self, taw):
        """
        mutates the chromosome via normal distribution
        """
        self.sigma = self.sigma * exp((-taw) * normalvariate(0, 1))

        mx = self.x + self.sigma + normalvariate(0, 1)
        if -512 <= mx <= 512:
            self.x = mx

        my = self.y + self.sigma + normalvariate(0, 1)
        if -512 <= my <= 512:
            self.y = my

    def fitness(self):
        """
        :return: fitness of chromosome.
        """
        return 1 / abs((-1) * (self.y + 47) * sin(sqrt(abs(self.y + self.x / 2 + 47))) - self.x * sin(
            sqrt(abs(self.x - (self.y + 47)))) + 959.6407)


def parent_selection(population: List[Chromosome], parents_num: int):
    """
    selects parents via fitness proportional selection - round weal
    :param population: current population
    :param parents_num: number of parents we need
    :return: parents
    """
    parents = []

    fitnesses = []
    for p in population:
        fitnesses.append(p.fitness())

    _sum = sum(fitnesses)
    if _sum < 1:  # all fitnesses are 0
        parents = population
        parents.extend(population)
        return parents

    # create sum fitness array
    sum_fitnesses = []
    sum_fitness = 0
    for i in range(len(fitnesses)):
        sum_fitness += fitnesses[i]
        sum_fitnesses.append(sum_fitness)

    # choose parents due to their fitness
    i = 0
    while i < parents_num:
        rand = randint(1, int(_sum))
        for j in range(len(sum_fitnesses)):
            if rand < sum_fitnesses[j]:
                parents.append(population[j])
                i += 1
                break

    return parents


def child_selection(population: list, tournament_size: int):
    """
    selects children from population via Q tournament procedure
    :param population: current population
    :return: children array
    """
    children = []
    tournament_number = int(len(population) / tournament_size)
    for k in range(tournament_number):
        best_fitness = -1  # minimum number for this variable
        best_chromosome: Chromosome
        for i in range(tournament_size):
            p_index = randrange(0, len(population))
            selected: Chromosome = population[p_index]

            f = selected.fitness()
            if f > best_fitness:
                best_fitness = f
                best_chromosome = selected
            population.remove(selected)

        # noinspection PyUnboundLocalVariable
        children.append(best_chromosome)

    return children


def crossover(parents: List[Chromosome], population_size: int, pc: float):
    """
    performs crossover operation
    :param parents: parents array
    :param population_size: size of population we need to generate
    :return: generated generation
    """
    new_generation = []

    for i in range(population_size):
        p = uniform(0, 1)
        a, b = parents[i * 2], parents[i * 2 + 1]
        if p < pc:
            new_generation.append(
                Chromosome(pc * a.x + (1 - pc) * b.x, pc * a.y + (1 - pc) * b.y, pc * a.sigma + (1 - pc) * b.sigma))
            new_generation.append(
                Chromosome((1 - pc) * a.x + pc * b.x, (1 - pc) * a.y + pc * b.y, (1 - pc) * a.sigma + pc * b.sigma))
        else:
            new_generation.append(a)
            new_generation.append(b)

    return new_generation


def mutation(population: List[Chromosome], taw: float):
    """
    mutates chromosomes due to taw
    """
    for i in range(len(population)):
        population[i].mutate(taw)


def main():
    """
    the main function
    """
    # population array
    population = []

    # problem constants
    cal_fitness_num = 10000
    population_size = 100
    parents_num = 200  # number of parents in each parents selection
    tournament_size = 3
    pc = 0.8  # crossover probability
    taw = 0.5

    # fitnesses values
    fitnesses = []

    # initializing population
    for i in range(population_size):
        population.append(Chromosome(randint(-512, 512), randint(-512, 512), 1))

    # evaluation
    for k in range(int(cal_fitness_num / population_size)):
        # parents selection
        parents = parent_selection(population, parents_num)
        new_generation = crossover(parents, population_size, pc)  # crossover
        mutation(new_generation, taw)  # mutation
        new_generation.extend(population)  # mu + lambda
        # children selection for mu + lambda
        population = child_selection(new_generation, tournament_size)

        # calculating fitnesses
        fitnesses.clear()
        for p in population:
            fitnesses.append(p.fitness())

        print("\nGeneration #" + str(k + 1) + ":")
        print("Best Fitness: " + str(max(fitnesses)))
        print("Worst Fitness: " + str(min(fitnesses)))
        print("Average Fitness: " + str(sum(fitnesses) / len(fitnesses)))

    # find best chromosome
    best_chromosome = population[0]
    best_fitness = max(fitnesses)
    for p in population:
        best_chromosome_fitness = best_chromosome.fitness()
        if best_chromosome_fitness == best_fitness:
            break
        if p.fitness() > best_chromosome_fitness:
            best_chromosome = p

    # printing results
    print("\n\nStop Condition Reached!")
    print("The Best Chromosome Genome in the Last Generation is " + str(best_chromosome.x) + "  " +
          str(best_chromosome.y))


if __name__ == '__main__':
    main()

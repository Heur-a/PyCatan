import random
import numpy as np
import os
import time
import concurrent.futures
import csv
from deap import base, creator, tools, algorithms
from Agents.GeneticAgent import GeneticAgent
from experiment_base import simulate_match

# Configuració
POPULATION_SIZE = 30
GENERATIONS = 50
CXPB = 0.7  # Probabilitat de creuament
MUTPB = 0.2  # Probabilitat de mutació
NGENES = 12  # Nombre de gens al cromosoma
MATCHES_PER_AGENT = 1000  # Partides per avaluació

# Inicialitzar estructura DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NGENES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate_agent(individual):
    """Avaluar un agent amb múltiples partides"""
    agent = GeneticAgent(0, chromosome=individual)
    total_score = 0

    for _ in range(MATCHES_PER_AGENT):
        for position in range(4):  # Provar en totes les posicions
            _, points, rank = simulate_match(position, agent)
            # Fitness: 60% punts, 30% victòries, 10% posició
            fitness = 0.6 * points + 0.3 * (1 if rank == 1 else 0) + 0.1 * (5 - rank)
            total_score += fitness
    print(total_score / (MATCHES_PER_AGENT * 4))
    return total_score / (MATCHES_PER_AGENT * 4),


toolbox.register("evaluate", evaluate_agent)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Creuament combinat
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    # Configuració multiprocess
    pool = concurrent.futures.ProcessPoolExecutor()
    toolbox.register("map", pool.map)

    # Inicialitzar població
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Executar algorisme genètic
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS,
        stats=stats, halloffame=hof, verbose=True
    )

    # Guardar millor cromosoma
    best_agent = GeneticAgent(0, chromosome=hof[0])
    print(f"\nMillor cromosoma: {hof[0]}")
    print(f"Fitness: {hof[0].fitness.values[0]}")

    # Guardar resultats en CSV
    with open("genetic_training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Avg Fitness", "Min Fitness", "Max Fitness"])
        for gen, record in enumerate(log):
            writer.writerow([gen, record["avg"], record["min"], record["max"]])

    # Guardar cromosoma òptim
    with open("best_chromosome.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(hof[0])

    pool.shutdown()
    return hof[0]


if __name__ == "__main__":
    best_chromosome = main()
import random
import traceback

from Agents.GeneticAgent import GeneticAgent
from Agents.RandomAgent import RandomAgent
from Managers.GameDirector import GameDirector
from experiment_base import simulate_match


class Genethics:
    def __init__(self, pop, ngenes, prob_cross, prob_mut, max_gene_number):
        self.agent_scores = None
        self.ngenes = ngenes
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.pop = pop
        self.max_gene_number = max_gene_number
        self.chromosomes = self.init_population()

    def init_population(self):
        population = []
        for i in range(self.pop):
            chromosome = []
            for j in range(self.ngenes):
                chromosome.append(random.randint(0, self.max_gene_number))
            population.append(chromosome)

        print("population size: ", len(population))
        print("pop_sample: ", population[len(population) - 1])
        self.chromosomes = population
        return population

    @staticmethod
    def crossover(parent1, parent2, crosspoint):
        crosspoint = crosspoint - 1
        offspring = parent1[0:crosspoint]
        offspring.extend(parent2[crosspoint: len(parent2)])
        return offspring

    @staticmethod
    def swap_mutation(chromosome):
        index_a = random.randint(0, len(chromosome) - 1)
        index_b = random.randint(0, len(chromosome) - 1)
        temp = chromosome[index_a]
        chromosome[index_a] = chromosome[index_b]
        chromosome[index_b] = temp
        return chromosome

    @staticmethod
    def evaluate_agent(individual, MATCHES_PER_AGENT=10):
        """Avaluar un agent amb múltiples partides"""
        AgentClass = GeneticAgent.with_chromosome(individual)
        total_score = 0

        for _ in range(MATCHES_PER_AGENT):
            for position in range(4):  # Provar en totes les posicions
                _, points, rank = simulate_match(position, AgentClass)
                # Fitness: 60% punts, 30% victòries, 10% posició
                fitness = 0.6 * points + 0.3 * (1 if rank == 1 else 0) + 0.1 * (5 - rank)
                total_score += fitness
        print(total_score / (MATCHES_PER_AGENT * 4))
        return total_score / (MATCHES_PER_AGENT * 4),

    def evaluate(self):
        agent_scores = []
        for index, chromosome in enumerate(self.chromosomes):
            score = Genethics.evaluate_agent(chromosome)
            info = {"index": index, "score": score}
            agent_scores.append(info)
        agent_scores.sort(key=lambda x: x["score"])
        self.agent_scores = agent_scores

        return agent_scores

    def roulette_wheel_selection(self):
        # Calculate total fitness
        total_fitness = sum(individual["score"][0] for individual in self.agent_scores)

        # Calculate selection probabilities
        selection_probs = [individual["score"][0] / total_fitness for individual in self.agent_scores]

        # Select parents using roulette wheel
        selected_indices = []
        for _ in range(self.pop):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected_indices.append(i)
                    break

        return selected_indices

    def evolve(self):
        # Evaluate current population
        self.evaluate()

        # Select parents using roulette wheel
        selected_indices = self.roulette_wheel_selection()

        # Create new generation
        new_generation = []
        for i in range(0, self.pop, 2):
            # Get two parents
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[i + 1] if i + 1 < len(selected_indices) else selected_indices[i]

            parent1 = self.chromosomes[parent1_idx]
            parent2 = self.chromosomes[parent2_idx]

            # Crossover
            if random.random() < self.prob_cross:
                cross_point = random.randint(1, self.ngenes - 1)
                child1 = self.crossover(parent1, parent2, cross_point)
                child2 = self.crossover(parent2, parent1, cross_point)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            if random.random() < self.prob_mut:
                child1 = self.swap_mutation(child1)
            if random.random() < self.prob_mut:
                child2 = self.swap_mutation(child2)

            new_generation.extend([child1, child2])

        # Complete generational replacement
        self.chromosomes = new_generation[:self.pop]  # Ensure population size stays constant
        return self.chromosomes


if __name__ == "__main__":
    gens = Genethics(100, 12, 0.5, 0.1, 256)
    pops = gens.init_population()

    parent_1 = pops[0]
    parent_2 = pops[1]
    crosspoint_ = 2
    offspring_ = Genethics.crossover(parent_1, parent_2, crosspoint_)
    print("parent_1 is ", parent_1)
    print("parent_2 is ", parent_2)
    print("offspring: ", offspring_)

    swapped = Genethics.swap_mutation(offspring_)
    print("swapped: ", swapped)

    # Test evolution
    new_pop = gens.evolve()
    print("New population size:", len(new_pop))
    print("Sample of new population:", new_pop[0])
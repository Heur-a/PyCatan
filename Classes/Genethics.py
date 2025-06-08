import random


class Genethics:
    """Genethics implements a basic genetic algorithm with elitism.

    This class simulates the evolution of a population of chromosomes through
    selection, crossover, and mutation, aiming to optimize a fitness function.
    """

    def __init__(self, pop, ngenes, prob_cross, prob_mut, max_gene_number):
        """Initializes the Genethics genetic algorithm parameters.

        Args:
            pop (int): Number of chromosomes in the population.
            ngenes (int): Number of genes per chromosome.
            prob_cross (float): Probability of applying crossover.
            prob_mut (float): Probability of applying mutation.
            max_gene_number (int): Maximum value for each gene.
        """
        self.ngenes = ngenes
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.pop = pop
        self.max_gene_number = max_gene_number
        self.chromosomes = self.init_population()
        self.elite_chromosome = self.chromosomes[0]
        self.elite_score = 0

    def init_population(self):
        """Initializes the population with random chromosomes.

        Returns:
            list[list[int]]: A list of randomly initialized chromosomes.
        """
        return [[random.randint(0, self.max_gene_number) for _ in range(self.ngenes)]
                for _ in range(self.pop)]

    @staticmethod
    def crossover(parent1, parent2, crosspoint):
        """Performs one-point crossover between two parent chromosomes.

        Args:
            parent1 (list[int]): First parent chromosome.
            parent2 (list[int]): Second parent chromosome.
            crosspoint (int): The crossover point (1-based index).

        Returns:
            list[int]: The resulting child chromosome.
        """
        crosspoint = crosspoint - 1
        return parent1[0:crosspoint] + parent2[crosspoint:]

    @staticmethod
    def swap_mutation(chromosome):
        """Performs swap mutation by exchanging two random gene positions.

        Args:
            chromosome (list[int]): A chromosome to mutate.

        Returns:
            list[int]: The mutated chromosome.
        """
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def gen_mutation(self, chromosome):
        """Performs gene mutation by assigning a new random value to one gene.

        Args:
            chromosome (list[int]): A chromosome to mutate.

        Returns:
            list[int]: The mutated chromosome.
        """
        idx = random.randint(0, len(chromosome) - 1)
        chromosome[idx] = random.randint(0, self.max_gene_number)
        return chromosome

    def calculate_stats(self, scores):
        """Calculates statistical data from a list of fitness scores.

        Args:
            scores (list[float]): Fitness scores of the population.

        Returns:
            tuple: A tuple containing (average, minimum, maximum, standard deviation).
        """
        avg = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        std_dev = (sum((x - avg) ** 2 for x in scores) / len(scores)) ** 0.5
        return avg, min_score, max_score, std_dev

    def roulette_wheel_selection(self, scores):
        """Selects individuals based on fitness-proportional probability.

        Scores are shifted to be strictly positive to ensure valid probabilities.

        Args:
            scores (list[float]): Fitness scores of the population.

        Returns:
            list[int]: Indices of selected chromosomes.
        """
        min_score = min(scores)
        epsilon = 1e-6
        shifted_scores = [s - min_score + epsilon for s in scores]  # Ensure all > 0

        total_fitness = sum(shifted_scores)
        selection_probs = [s / total_fitness for s in shifted_scores]
        selected_indices = random.choices(range(len(scores)), weights=selection_probs, k=self.pop)
        return selected_indices

    def evolve(self, scores):
        """Evolves the current population to the next generation.

        Applies elitism to retain the best solution, performs roulette wheel selection,
        crossover, and both swap and gene mutation to generate new chromosomes.

        Args:
            scores (list[float]): Fitness scores of the current population.

        Returns:
            list[list[int]]: The new population of chromosomes.
        """
        selected_indices = self.roulette_wheel_selection(scores)
        new_generation = []

        # Elitism: preserve the best chromosome across generations
        elite_idx_score = scores.index(max(scores))
        if self.elite_score < scores[elite_idx_score]:
            self.elite_score = scores[elite_idx_score]
            self.elite_chromosome = self.chromosomes[elite_idx_score]

        # Create the rest of the population
        while len(new_generation) < self.pop:
            i1, i2 = random.sample(selected_indices, 2)
            parent1, parent2 = self.chromosomes[i1], self.chromosomes[i2]

            if random.random() < self.prob_cross:
                cross_point = random.randint(1, self.ngenes - 1)
                child1 = self.crossover(parent1, parent2, cross_point)
                child2 = self.crossover(parent2, parent1, cross_point)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < self.prob_mut:
                child1 = self.swap_mutation(child1)
            if random.random() < self.prob_mut:
                child2 = self.swap_mutation(child2)
            if random.random() < self.prob_mut:
                child1 = self.gen_mutation(child1)
            if random.random() < self.prob_mut:
                child2 = self.gen_mutation(child2)

            new_generation.extend([child1, child2])

        # Insert elite chromosome at the beginning
        new_generation.insert(0, self.elite_chromosome)

        self.chromosomes = new_generation[:self.pop]
        return self.chromosomes

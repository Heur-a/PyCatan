"""Train a Genetic Algorithm to optimize agent performance using simulated matches.

This script initializes and trains a genetic algorithm using the Genethics class.
Each individual (chromosome) is evaluated through multiple simulations. The algorithm
evolves until convergence based on average fitness difference between generations.

Results (statistics and best agents) are saved to CSV files.
"""

import csv
import multiprocessing
import os
import time
import traceback
from datetime import datetime
from functools import partial

from Agents.AdrianHerasAgent import AdrianHerasAgent as aa
from Agents.AlexElenaMarcosGeneticAgent import AlexElenaMarcosGeneticAgent
from Classes.Genethics import Genethics
from Agents.RandomAgent import RandomAgent as ra
from Managers.GameDirector import GameDirector
from Agents.SigmaAgent import SigmaAgent as sa


def evaluate_agent_wrapper(individual, matches_per_agent=10):
    """Evaluates the performance of an individual agent based on simulations.

    This function wraps the simulation logic for multiprocessing. It evaluates
    the given chromosome across several matches and computes a fitness score.

    Args:
        individual (list[int]): The chromosome representing the agent.
        matches_per_agent (int): Number of full games to simulate per individual.

    Returns:
        float: Average fitness score across all simulations.
    """
    try:
        AgentClass = AlexElenaMarcosGeneticAgent.with_chromosome(individual)
        total_score = 0

        for _ in range(matches_per_agent):
            for position in range(4):  # Test in all player positions
                _, points, rank = simulate_match(position, AgentClass)
                fitness = 0.6 * points + 0.3 * (1 if rank == 1 else 0) + 0.1 * (5 - rank)
                total_score += fitness

        return total_score / (matches_per_agent * 4)
    except Exception as e:
        traceback.print_exc()
        return 0  # Return zero fitness on error


class GeneticTrainer:
    """Trainer class for managing the genetic algorithm training process."""

    def __init__(self, config):
        """Initializes the genetic trainer with the given configuration.

        Args:
            config (dict): Configuration dictionary with keys such as:
                - 'pop_size': Population size
                - 'num_genes': Number of genes per individual
                - 'crossover_prob': Crossover probability
                - 'mutation_prob': Mutation probability
                - 'max_gene_value': Maximum gene value
                - 'num_processes': Number of parallel processes
                - 'experiment_name': Optional experiment name
        """
        self.config = config
        self.ga = Genethics(
            pop=config['pop_size'],
            ngenes=config['num_genes'],
            prob_cross=config['crossover_prob'],
            prob_mut=config['mutation_prob'],
            max_gene_number=config['max_gene_value']
        )
        self.num_processes = config.get('num_processes', multiprocessing.cpu_count())
        self.experiment_name = config.get('experiment_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(f"results/{self.experiment_name}", exist_ok=True)

        # Initialize CSV files for logging
        self.stats_file = f"results/{self.experiment_name}/stats.csv"
        self.best_agents_file = f"results/{self.experiment_name}/best_agents.csv"

        with open(self.stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'timestamp', 'avg_fitness', 'min_fitness',
                             'max_fitness', 'std_dev', 'time_sec'])

        with open(self.best_agents_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'fitness', 'chromosome'])

    def evaluate_population(self, matches_per_agent=10):
        """Evaluates the entire population in parallel.

        Args:
            matches_per_agent (int): Number of matches per individual.

        Returns:
            list[float]: Fitness scores for each chromosome in the population.
        """
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            evaluate_func = partial(evaluate_agent_wrapper, matches_per_agent=matches_per_agent)
            scores = pool.map(evaluate_func, self.ga.chromosomes)
        return scores

    def log_generation_stats(self, generation, start_time, scores):
        """Logs statistics and best individual of a generation.

        Args:
            generation (int): Current generation number.
            start_time (float): Timestamp when the generation started.
            scores (list[float]): Fitness scores of the current generation.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_elapsed = time.time() - start_time
        avg, min_score, max_score, std_dev = self.ga.calculate_stats(scores)

        # Save overall statistics
        with open(self.stats_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generation, timestamp, avg, min_score, max_score, std_dev, time_elapsed])

        # Save best individual
        best_index = scores.index(max(scores))
        best_chromosome = self.ga.chromosomes[best_index]
        with open(self.best_agents_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generation, scores[best_index], ';'.join(map(str, best_chromosome))])

        print(f"\nGeneration {generation} completed in {time_elapsed:.2f} seconds")
        print(f"  Avg: {avg:.4f} | Min: {min_score:.4f} | Max: {max_score:.4f} | Std: {std_dev:.4f}")

    def train(self):
        """Main training loop for the genetic algorithm.

        Repeats evaluation, logging, and evolution until the convergence condition is met.
        """
        print(f"\n🚀 Starting experiment: {self.experiment_name}")
        print(f"  Population: {self.config['pop_size']} | Genes: {self.config['num_genes']}")
        print(f"  Crossover Prob.: {self.config['crossover_prob']} | Mutation Prob.: {self.config['mutation_prob']}")
        print(f"  Processes: {self.num_processes}\n")

        convergence_threshold = self.config.get('convergence', 1e-2)
        prev_avg = None
        gen = 1

        while True:
            start_time = time.time()

            # Evaluate current population
            scores = self.evaluate_population(self.config.get('matches_per_agent', 10))

            # Log generation statistics
            self.log_generation_stats(gen, start_time, scores)

            avg, _, _, _ = self.ga.calculate_stats(scores)

            # Check for convergence
            if prev_avg is not None:
                avg_diff = avg - prev_avg
                print(f"  Avg. difference from previous generation: {avg_diff:.6f}")
                if convergence_threshold > abs(avg_diff):
                    print(f"\n🧬 Convergence reached at generation {gen} with avg. diff {avg_diff:.6f}")
                    break

            # Evolve population
            self.ga.evolve(scores)
            prev_avg = avg
            gen += 1

        print(f"\n✅ Experiment completed! Results saved in: results/{self.experiment_name}")


def run_genetic_algorithm(config):
    """Runs the full genetic algorithm training with the provided configuration.

    Args:
        config (dict): Configuration dictionary for the trainer.
    """
    trainer = GeneticTrainer(config)
    trainer.train()

def simulate_match(position, agente_alumno):
    try:
        match_agents = [aa, ra, sa]
        match_agents.insert(position, agente_alumno)

        game_director = GameDirector(agents=match_agents, max_rounds=200, store_trace=False)
        game_trace = game_director.game_start(print_outcome=False)

        last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
        last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
        victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

        agent_id = f"J{position}"
        points = int(victory_points[agent_id])
        winner = max(victory_points, key=lambda player: int(victory_points[player]))
        victory = 1 if winner == agent_id else 0

        ordenados = sorted(victory_points.items(), key=lambda item: int(item[1]), reverse=True)
        for idx, (jugador, _) in enumerate(ordenados, start=1):
            if jugador == agent_id:
                rank = idx
                break
        return (victory, points, rank)
    except Exception as e:
        print("‼️ Ha ocorregut una excepció en simulate_match:")
        print("Tipus d'excepció:", type(e).__name__)
        print("Missatge:", str(e))
        print("Traça completa:")
        traceback.print_exc()
        return (0, 0, 4)


if __name__ == "__main__":
    # Ask for experiment name
    experiment_name = input("🔬 Enter experiment name: ").strip()
    if not experiment_name:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Define experiment configuration
    config = {
        'experiment_name': experiment_name,
        'pop_size': 20,             # de 20 a 100 funciona bien
        'num_genes': 12,            # Invariable
        'crossover_prob': 0.5,      # de 0 a 1
        'mutation_prob': 0.1,       # más de 0.4 puede ser inviable
        'max_gene_value': 256,      # Se puede variar pero no afecta mucho
        'convergence': 3e-2,        # 0.03 es un buen punto encontrado
        'matches_per_agent': 10,    # Bien para acabar rápido
        'num_processes': 18         # 18 máximo antes de que te explote el ordenador
    }

    run_genetic_algorithm(config)

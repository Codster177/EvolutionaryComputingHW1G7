import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import sys

def evalOneMax(individual):
    return (sum(individual),)

def selRank(individuals, k):
    sorted_pop = sorted(individuals, key=lambda x: x.fitness.values[0])
    return random.choices(sorted_pop, weights=range(1, len(sorted_pop) + 1), k=k)

selTournament = lambda pop, k: tools.selTournament(pop, k, tournsize=2)

# methodName = sys.argv[1] if len(sys.argv) > 1 else None

# if methodName == "roulette":
#     method = tools.selRoulette
# elif methodName == "tournament":
#     method = selTournament
# elif methodName == "rank":
#     method = selRank
# elif methodName == "random":
#     method = tools.selRandom
# else:
#     print("Options: python3 onemax-q5.py <selection-method>")
#     print("   'roulette' - Fitness proportional")
#     print("   'tournament' - Tournament selection")
#     print("   'rank' - Rank selection")
#     print("   'random' - Random selection")
#     sys.exit()

# Parameters
INDIVIDUAL = 200
POPULATION = 200
GENERATIONS = 200
CROSSOVER = 0.9
MUTATION_RATE = 0.002
MUTPB = 1.0

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


print("Configuration:")
print(f"  Population size: {POPULATION}")
print(f"  Generations: {GENERATIONS}")
print(f"  Crossover probability: {CROSSOVER}")
print(f"  Mutation probability: {MUTPB}")
print(f"  Individual size: {INDIVIDUAL} bits")
print(f"  Optimal fitness: {INDIVIDUAL}")

def single_run(seed, method):
    """Runs the one max algorithm and stores the results"""
    random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, INDIVIDUAL)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("select", method)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_RATE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)    # Average fitness
    stats.register("max", np.max)     # Best fitness

    population = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(1)

    final_population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CROSSOVER,
        mutpb=MUTPB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=False
    )
    
    # Get the data
    generation_max = logbook.select("max")
    generation_avg = logbook.select("avg")
    best = hof[0].fitness.values[0]
    
    # Find generation where optimal was first found
    generation_optimal = None
    for gen, best_fit in enumerate(generation_max):
        if best_fit >= INDIVIDUAL:
            generation_optimal = gen
            break
    
    return {
        'generation_max': generation_max,
        'generation_avg': generation_avg,
        'best': best,
        'generation_optimal': generation_optimal
    }

NUM_RUNS = 50

def gen_results(method):

    all_results = []

    print(f"Running {NUM_RUNS} experiments...")
    for i in range(NUM_RUNS):
        result = single_run(seed=i, method=method)
        all_results.append(result)
        print(f"Run {i+1}/{NUM_RUNS} complete - Best: {result['best']}")
    return all_results

methods = [
    ("Rank", selRank),
    # ("Tournament", selTournament),
    # ("Fitness-proportional", tools.selRoulette),
    # ("Random", tools.selRandom),
]
results = list(map(lambda t: (t[0], gen_results(t[1])), methods))

print("="*60)
print("All experiments complete!")

# Plotting
plt.figure(figsize=(12, 7))

colors = ['#fa5252', '#be4bdb', '#228be6', '#40c057']
color_idx = 0

for k, all_results in results:
    best_fit_per_gen = np.array([r['generation_max'] for r in all_results])
    avg_fit_per_gen = np.array([r['generation_avg'] for r in all_results])
    overall_best_fit = np.array([r['best'] for r in all_results])
    optimal_gens = [r['generation_optimal'] for r in all_results if r['generation_optimal'] is not None]

    best_mean = np.mean(overall_best_fit)
    best_std = np.std(overall_best_fit, ddof=1)
    best_ci = 1.96 * best_std / np.sqrt(NUM_RUNS)

    print("="*60)
    print(f"Selection method:    {k}")
    print(f"Mean:                {best_mean:.2f}")
    print(f"Standard Deviation:  {best_std:.2f}")
    print(f"95% CI:              [{best_mean - best_ci:.2f}, {best_mean + best_ci:.2f}]")

    if len(optimal_gens) > 0:
        gen_mean = np.mean(optimal_gens)
        gen_std = np.std(optimal_gens, ddof=1)
        gen_ci = 1.96 * gen_std / np.sqrt(len(optimal_gens))
        print(f"\nGeneration Optimal Found ({len(optimal_gens)} out of {NUM_RUNS} runs):")
        print(f"  Mean:                {gen_mean:.2f}")
        print(f"  Standard Deviation:  {gen_std:.2f}")
        print(f"  95% CI:              [{gen_mean - gen_ci:.2f}, {gen_mean + gen_ci:.2f}]")
    else:
        print(f"No runs found the optimal individual")

    best_per_gen_mean = np.mean(best_fit_per_gen, axis=0)
    best_per_gen_std = np.std(best_fit_per_gen, axis=0, ddof=1)
    best_per_gen_ci = 1.96 * best_per_gen_std / np.sqrt(NUM_RUNS)

    mean_avg_per_gen = np.mean(avg_fit_per_gen, axis=0)
    std_avg_per_gen = np.std(avg_fit_per_gen, axis=0, ddof=1)
    ci_avg_per_gen = 1.96 * std_avg_per_gen / np.sqrt(NUM_RUNS)

    generations = list(range(len(best_per_gen_mean)))

    # Get color for this selection method
    color = colors[color_idx]
    color_idx += 1

    # Best fit plot (1a)
    plt.plot(generations, best_per_gen_mean, '-', color=color, label=f'{k} - Best Fitness', linewidth=2)
    plt.fill_between(generations,
                    best_per_gen_mean - best_per_gen_ci,
                    best_per_gen_mean + best_per_gen_ci,
                    color=color, alpha=0.2)

    plt.plot(generations, mean_avg_per_gen, '--', color=color, label=f'{k} - Mean Fitness', linewidth=2)
    plt.fill_between(generations,
                    mean_avg_per_gen - ci_avg_per_gen,
                    mean_avg_per_gen + ci_avg_per_gen,
                    color=color, alpha=0.1)

plt.axhline(y=INDIVIDUAL, color='r', linestyle='--', label=f'Optimal ({INDIVIDUAL})', linewidth=1)

plt.xlabel('Generation', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.title(f'GA Performance on OneMax Problem (Tuned)\nAveraged over {NUM_RUNS} runs', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
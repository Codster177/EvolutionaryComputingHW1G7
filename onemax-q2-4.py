import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

INDIVIDUAL = 200
POPULATION = [100, 50, 200]
GENERATIONS = 100
CROSSOVER = [0.8, 0.4, 1.0]
MUTATION_RATE = [0.005, 0.001, 0.01]
MUTPB = 1.0

def evalOneMax(individual):
    return sum(individual),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def single_run(seed, population_val, crossover_val, mutation_rate_val):
    random.seed(seed)
    

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, INDIVIDUAL)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_rate_val)

    population = toolbox.population(n=population_val)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    final_population, log = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_val,
        mutpb=MUTPB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    generation_max = log.select("max")
    generation_avg = log.select("avg")
    best = hof[0].fitness.values[0]
    
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

def run_onemax(parameters):
    all_results = []
    
    population_val = parameters[0]
    crossover_val = parameters[1]
    mutation_rate_val = parameters[2]

    print(f"Running {NUM_RUNS} experiments...")
    for i in range(NUM_RUNS):
        result = single_run(i, population_val, crossover_val, mutation_rate_val)
        all_results.append(result)
        print(f"Run {i+1}/{NUM_RUNS} complete - Best: {result['best']}")


    return all_results

print(f"Baseline configuration:")
print(f"  Population size: {POPULATION[0]}")
print(f"  Generations: {GENERATIONS}")
print(f"  Crossover probability: {CROSSOVER[0]}")
print(f"  Mutation probability: {MUTPB}")
print(f"  Bit Mutation Rate: {MUTATION_RATE[0]}")
print(f"  Individual size: {INDIVIDUAL} bits")
print(f"  Optimal fitness: {INDIVIDUAL}")

question_num = input(f"Enter question number (2, 3, or 4): ")

question_results = []
test_values = []

if (question_num == "2"):
    pop_val = int(input(f"Enter unique value for population size: "))
    POPULATION.append(pop_val)
    for population_val in POPULATION:
        test_values.append(f"Population size - {population_val}")
        print(f"\n\nRunning test with population size: {population_val}\n")
        question_results.append(run_onemax([population_val, CROSSOVER[0], MUTATION_RATE[0]]))

elif (question_num == "3"):
    mutation_val = float(input("Enter unique value for mutation rate: "))
    MUTATION_RATE.append(mutation_val)
    for mutation_rate_val in MUTATION_RATE:
        test_values.append(f"Mutation rate - {mutation_rate_val}")
        print(f"\n\nRunning test with mutation rate: {mutation_rate_val}\n")
        question_results.append(run_onemax([POPULATION[0], CROSSOVER[0], mutation_rate_val]))

elif (question_num == "4"):
    crossover_val = float(input("Enter unique value for crossover rate: "))
    CROSSOVER.append(crossover_val)
    for crossover_rate_val in CROSSOVER:
        test_values.append(f"Crossover rate - {crossover_rate_val}")
        print(f"\n\nRunning test with crossover rate: {crossover_rate_val}\n")
        question_results.append(run_onemax([POPULATION[0], crossover_rate_val, MUTATION_RATE[0]]))


print("="*60)
print("All experiments complete!")

plt.figure(figsize=(12, 7))

colors = ['#fa5252', '#be4bdb', '#228be6', '#40c057']
color_idx = 0

for index, all_results in enumerate(question_results):
    best_fit_per_gen = np.array([r['generation_max'] for r in all_results])
    avg_fit_per_gen = np.array([r['generation_avg'] for r in all_results])
    overall_best_fit = np.array([r['best'] for r in all_results])
    optimal_gens = [r['generation_optimal'] for r in all_results if r['generation_optimal'] is not None]

    best_mean = np.mean(overall_best_fit)
    best_std = np.std(overall_best_fit, ddof=1)
    best_ci = 1.96 * best_std / np.sqrt(NUM_RUNS)

    print("="*60)

    print(f"Test Value:          {test_values[index]}")
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

    color = colors[color_idx]
    color_idx += 1

    plt.plot(generations, best_per_gen_mean, '-', color=color, label=f'Best Fitness: {test_values[index]}', linewidth=2)
    plt.fill_between(generations, 
                    best_per_gen_mean - best_per_gen_ci, 
                    best_per_gen_mean + best_per_gen_ci, 
                    color=color, alpha=0.2)

    plt.plot(generations, mean_avg_per_gen, '--', color=color,  label=f'Mean Fitness: {test_values[index]}', linewidth=2)
    plt.fill_between(generations, 
                    mean_avg_per_gen - ci_avg_per_gen, 
                    mean_avg_per_gen + ci_avg_per_gen, 
                    color=color, alpha=0.1)
    

plt.axhline(y=INDIVIDUAL, color='r', linestyle='--', label=f'Optimal ({INDIVIDUAL})', linewidth=1)

plt.xlabel('Generation', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.title('GA Performance on OneMax Problem\nAveraged over 50 runs', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
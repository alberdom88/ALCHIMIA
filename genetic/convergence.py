import re
import pandas as pd
import numpy as np
import sys

# ============================================
# CRITERI DI CONVERGENZA - 4 CRITERIA (3/4 REQUIRED)
# ============================================
# 1. BEST SCORE STAGNATION: nessun nuovo minimo assoluto per N generazioni
# 2. MEAN SCORE STAGNATION: media non migliora per N generazioni
# 3. POPULATION DECLINE: popolazione < X% del massimo per N generazioni (UPDATED!)
# 4. MINIMUM GENERATION: generazione minima prima di poter fermarsi
# STOP quando 3 su 4 criteri sono soddisfatti
# ============================================

# Input/output files
input_file = sys.argv[1]
output_csv = input_file.replace('.txt', '_convergence.csv')

# Parameters
max_gens_without_new_min = 25         # Criterio 1: nessun nuovo minimo best
max_gens_without_mean_improve = 25    # Criterio 2: media non migliora
max_gens_below_pop_threshold = 25     # Criterio 3: popolazione bassa per N gen (UPDATED!)
mean_improvement_threshold = 0.1      # Soglia miglioramento media
population_decline_threshold = 0.50   # Soglia popolazione (50% del max) (UPDATED!)
population_threshold = 100            # Soglia popolazione 
min_generation = 100                  # Generazione minima

def parse_file(filename):
    """Parse the genetic algorithm output file"""
    data = []
    with open(filename) as f:
        for line in f:
            parts = line.strip().split()
            gen_match = re.search(r'GEN(\d+)', line)
            if gen_match and len(parts) >= 10:
                gen = int(gen_match.group(1))
                try:
                    gen_idx = [i for i,p in enumerate(parts) if 'GEN' in p][0]
                    scores = [float(parts[i]) for i in range(gen_idx+2, min(gen_idx+8, len(parts)))]
                    if len(scores) >= 4:
                        data.append({
                            'gen': gen,
                            'score1': scores[0],
                            'score2': scores[1],
                            'score3': scores[2],
                            'score4': scores[3]
                        })
                except:
                    continue
    return pd.DataFrame(data)

print("=" * 70)
print("CONVERGENCE ANALYSIS - 4 CRITERIA (3/4 REQUIRED)")
print("=" * 70)

df = parse_file(input_file)

# Statistics per generation
grouped = df.groupby('gen').agg({
    'score1': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
grouped.columns = ['gen', 'mean_score', 'std_score', 'best_score', 'worst_score', 'count']

print(f"\nTotal molecules: {len(df)}")
print(f"Generations: {grouped['gen'].min()} to {grouped['gen'].max()}")
print(f"Minimum generation required: {min_generation}")

# ============================================
# CRITERION 1: BEST SCORE STAGNATION
# ============================================
grouped['global_best'] = grouped['best_score'].cummin()
grouped['is_new_best_min'] = grouped['best_score'] == grouped['global_best']

gens_since_new_best = []
last_new_best_gen = 0
for idx, row in grouped.iterrows():
    if row['is_new_best_min']:
        last_new_best_gen = row['gen']
        gens_since_new_best.append(0)
    else:
        gens_since_new_best.append(int(row['gen'] - last_new_best_gen))

grouped['gens_since_new_best'] = gens_since_new_best
grouped['crit1_best_stagnant'] = (grouped['gens_since_new_best'] >= max_gens_without_new_min).astype(int)

new_best_gens = grouped[grouped['is_new_best_min']]['gen'].values
last_new_best = new_best_gens[-1] if len(new_best_gens) > 0 else 0

# ============================================
# CRITERION 2: MEAN SCORE STAGNATION
# ============================================
mean_improvements = []
for idx, row in grouped.iterrows():
    if idx == 0:
        mean_improvements.append(True)
    else:
        improved = row['mean_score'] < (grouped.loc[:idx-1, 'mean_score'].min() - mean_improvement_threshold)
        mean_improvements.append(improved)

grouped['is_mean_improvement'] = mean_improvements

gens_since_new_mean = []
last_new_mean_gen = 0
for idx, row in grouped.iterrows():
    if grouped.loc[idx, 'is_mean_improvement']:
        last_new_mean_gen = row['gen']
        gens_since_new_mean.append(0)
    else:
        gens_since_new_mean.append(int(row['gen'] - last_new_mean_gen))

grouped['gens_since_new_mean'] = gens_since_new_mean
grouped['crit2_mean_stagnant'] = (grouped['gens_since_new_mean'] >= max_gens_without_mean_improve).astype(int)
grouped['global_best_mean'] = grouped['mean_score'].cummin()

mean_improve_gens = grouped[grouped['is_mean_improvement']]['gen'].values
last_mean_improve = mean_improve_gens[-1] if len(mean_improve_gens) > 0 else 0

# ============================================
# CRITERION 3: POPULATION DECLINE (UPDATED!)
# ============================================
# Find maximum population size
max_population = grouped['count'].max()
#population_threshold = max_population * population_decline_threshold


# Check if population is below threshold
grouped['is_below_threshold'] = ((grouped['count'] < population_threshold) & (grouped['gen'] > 5)).astype(int)

# Count consecutive generations below threshold
consecutive_below = []
count = 0
for idx, row in grouped.iterrows():
    if row['is_below_threshold']:
        count += 1
        consecutive_below.append(count)
    else:
        count = 0
        consecutive_below.append(0)

grouped['consecutive_below_threshold'] = consecutive_below
grouped['crit3_pop_decline'] = (grouped['consecutive_below_threshold'] >= max_gens_below_pop_threshold).astype(int)

# Find first generation that triggered criterion
first_pop_decline_gen = None
pop_decline_gens = grouped[grouped['crit3_pop_decline'] == 1]['gen'].values
if len(pop_decline_gens) > 0:
    first_pop_decline_gen = pop_decline_gens[0]

# ============================================
# DECISION LOGIC (3 out of 4 criteria + min gen)
# ============================================
grouped['criteria_met'] = (grouped['crit1_best_stagnant'] + 
                           grouped['crit2_mean_stagnant'] + 
                           grouped['crit3_pop_decline'])

gen_stop = None
for idx, row in grouped.iterrows():
    gen = int(row['gen'])
    # 3 out of 4 criteria (excluding min_gen from count) + minimum generation requirement
    if row['criteria_met'] >= 3 and gen >= min_generation:
        gen_stop = gen
        break

# Save results
output_cols = ['gen', 'mean_score', 'std_score', 'best_score', 'count',
               'global_best', 'is_new_best_min', 'gens_since_new_best',
               'global_best_mean', 'is_mean_improvement', 'gens_since_new_mean',
               'is_below_threshold', 'consecutive_below_threshold',
               'crit1_best_stagnant', 'crit2_mean_stagnant', 'crit3_pop_decline', 
               'criteria_met']
grouped[output_cols].to_csv(output_csv, index=False)

# ============================================
# PRINT SUMMARY
# ============================================
print("\n" + "=" * 70)
print("CONVERGENCE CRITERIA (3 OUT OF 4 REQUIRED)")
print("=" * 70)

current_gen = grouped['gen'].max()

print(f"\n1. BEST SCORE STAGNATION:")
print(f"   Threshold: No new best minimum for {max_gens_without_new_min} generations")
print(f"   Last new best minimum: GEN {last_new_best}")
current_best_stagnation = current_gen - last_new_best
print(f"   Current stagnation: {current_best_stagnation} generations")
print(f"   Status: {'✓ PASSED' if current_best_stagnation >= max_gens_without_new_min else f'✗ NOT MET ({max_gens_without_new_min - current_best_stagnation} more needed)'}")

new_best_history = grouped[grouped['is_new_best_min']][['gen', 'best_score']]
print(f"\n   History of new best minima ({len(new_best_history)} total):")
if len(new_best_history) <= 20:
    for _, row in new_best_history.iterrows():
        print(f"     GEN {int(row['gen']):3d}: score = {row['best_score']:.4f}")
else:
    print("   First 10:")
    for _, row in new_best_history.head(10).iterrows():
        print(f"     GEN {int(row['gen']):3d}: score = {row['best_score']:.4f}")
    print("   ...")
    print("   Last 10:")
    for _, row in new_best_history.tail(10).iterrows():
        print(f"     GEN {int(row['gen']):3d}: score = {row['best_score']:.4f}")

print(f"\n2. MEAN SCORE STAGNATION:")
print(f"   Threshold: No mean improvement > {mean_improvement_threshold} for {max_gens_without_mean_improve} generations")
print(f"   Last mean improvement: GEN {last_mean_improve}")
current_mean_stagnation = current_gen - last_mean_improve
print(f"   Current stagnation: {current_mean_stagnation} generations")
print(f"   Status: {'✓ PASSED' if current_mean_stagnation >= max_gens_without_mean_improve else f'✗ NOT MET ({max_gens_without_mean_improve - current_mean_stagnation} more needed)'}")

mean_improve_history = grouped[grouped['is_mean_improvement']][['gen', 'mean_score']]
print(f"\n   History of mean improvements ({len(mean_improve_history)} total):")
if len(mean_improve_history) <= 20:
    for _, row in mean_improve_history.iterrows():
        print(f"     GEN {int(row['gen']):3d}: mean = {row['mean_score']:.4f}")
else:
    print("   First 10:")
    for _, row in mean_improve_history.head(10).iterrows():
        print(f"     GEN {int(row['gen']):3d}: mean = {row['mean_score']:.4f}")
    print("   ...")
    print("   Last 10:")
    for _, row in mean_improve_history.tail(10).iterrows():
        print(f"     GEN {int(row['gen']):3d}: mean = {row['mean_score']:.4f}")

print(f"\n3. POPULATION DECLINE:")
print(f"   Threshold: Population < {population_decline_threshold*100:.0f}% of max ({int(population_threshold)}) for {max_gens_below_pop_threshold} consecutive generations")
print(f"   Maximum population: {int(max_population)}")
print(f"   Current population: {int(grouped.loc[grouped['gen'] == current_gen, 'count'].values[0])}")
current_consecutive_below = int(grouped.loc[grouped['gen'] == current_gen, 'consecutive_below_threshold'].values[0])
print(f"   Consecutive gens below threshold: {current_consecutive_below}")
if first_pop_decline_gen is not None:
    print(f"   First triggered at: GEN {first_pop_decline_gen}")
print(f"   Status: {'✓ PASSED' if current_consecutive_below >= max_gens_below_pop_threshold else f'✗ NOT MET ({max_gens_below_pop_threshold - current_consecutive_below} more needed)'}")

# Show population evolution
print(f"\n   Population evolution (last 20 gens):")
last_20 = grouped.tail(20)[['gen', 'count']]
for _, row in last_20.iterrows():
    below_marker = " ← BELOW" if row['count'] < population_threshold else ""
    print(f"     GEN {int(row['gen']):3d}: count = {int(row['count']):3d}{below_marker}")

print(f"\n4. MINIMUM GENERATION REQUIREMENT:")
print(f"   Minimum required: GEN {min_generation}")
print(f"   Current generation: GEN {current_gen}")
print(f"   Status: {'✓ PASSED' if current_gen >= min_generation else f'✗ NOT MET ({min_generation - current_gen} more needed)'}")

print(f"\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
if gen_stop is not None:
    print(f"\n✓ STOP AT GENERATION: {gen_stop}")
    print(f"  3 out of 4 criteria satisfied (+ min gen {min_generation})")

    stop_row = grouped[grouped['gen'] == gen_stop].iloc[0]
    print(f"\n  Stats at GEN {gen_stop}:")
    print(f"    Best score: {stop_row['best_score']:.3f}")
    print(f"    Global best: {stop_row['global_best']:.3f}")
    print(f"    Mean score: {stop_row['mean_score']:.3f}")
    print(f"    Population size: {int(stop_row['count'])} (threshold: {int(population_threshold)})")
    print(f"\n  Criteria status:")
    print(f"    - Criterion 1 (Best): {'✓' if stop_row['crit1_best_stagnant'] else '✗'} ({int(stop_row['gens_since_new_best'])} gens)")
    print(f"    - Criterion 2 (Mean): {'✓' if stop_row['crit2_mean_stagnant'] else '✗'} ({int(stop_row['gens_since_new_mean'])} gens)")
    print(f"    - Criterion 3 (Pop):  {'✓' if stop_row['crit3_pop_decline'] else '✗'} ({int(stop_row['consecutive_below_threshold'])} consecutive gens below)")
    print(f"    - Criterion 4 (Min):  ✓ (gen {gen_stop} >= {min_generation})")
else:
    max_row = grouped[grouped['gen'] == current_gen].iloc[0]
    print(f"\n⚠ CONVERGENCE NOT YET REACHED")
    print(f"  Current generation: {current_gen}")
    print(f"  Criteria met: {int(max_row['criteria_met'])}/3 (need 3)")

    if current_gen < min_generation:
        print(f"\n  ⚠ BLOCKED BY MINIMUM GENERATION REQUIREMENT")
        print(f"    Need to reach GEN {min_generation} ({min_generation - current_gen} more generations)")

    print(f"\n  Current status:")
    print(f"    - Criterion 1 (Best): {'✓' if max_row['crit1_best_stagnant'] else '✗'} ({int(max_row['gens_since_new_best'])} gens since new best)")
    print(f"    - Criterion 2 (Mean): {'✓' if max_row['crit2_mean_stagnant'] else '✗'} ({int(max_row['gens_since_new_mean'])} gens since mean improve)")
    print(f"    - Criterion 3 (Pop):  {'✓' if max_row['crit3_pop_decline'] else '✗'} ({int(max_row['consecutive_below_threshold'])} consecutive gens below threshold)")
    print(f"    - Criterion 4 (Min):  {'✓' if current_gen >= min_generation else '✗'} (current: {current_gen}, required: {min_generation})")
    print(f"\n  Suggestion: {'Stop soon' if max_row['criteria_met'] >= 2 and current_gen >= min_generation else 'Continue optimization'}")

print(f"\n" + "=" * 70)
print(f"Results saved to: {output_csv}")
print("=" * 70)

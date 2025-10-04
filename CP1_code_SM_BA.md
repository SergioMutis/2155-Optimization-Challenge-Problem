# CHALLENGE PROBLEM 1 - CODE+-
2.156 ML & AI for Design Engineering  
Berfin Ataman & Sergio Mutis

-----------------------------------------------------------------------------------


# Challenge Problem Instructions

In this challenge problem, the task is to design **2D planar linkage mechanisms** that can trace six given target curves (the “egg” paths). The optimization must balance **accuracy** (distance to target curve) and **efficiency** (total material used), while respecting complexity constraints.

### Constraints
Your solutions must satisfy all three of the following:
1. **Distance Constraint**: Mechanism’s traced curve must be within **0.75** distance of the target curve.  
2. **Material Constraint**: Total linkage length must be **≤ 10.0**.  
3. **Complexity Constraint**: Mechanisms may have at most **20 joints**.  

Mechanisms that violate these constraints will be discarded and will not count toward the score.

### Objective & Scoring
- Each mechanism is evaluated on **two objectives**:  
  - Distance to target curve (minimize).  
  - Material usage (minimize).  
- Performance is measured using the **hypervolume (HV)** indicator with reference point **(0.75, 10.0)**.  
- For each target curve, you may submit up to **1000 mechanisms**.  
- The **average hypervolume across all six target curves** determines your final score.

### Optimization Methods
You are expected to use and possibly combine the following strategies:
- **Genetic Algorithms (GA):** Optimize both mechanism structures (edges, connectivity) and joint positions. You may experiment with:
  - Preprocessing or filtering random mechanisms before GA.  
  - Modified mutation and crossover operators.  
  - Mixed-variable GA setups using connectivity matrices.  
- **Gradient-Based Optimization (GD):** Use differentiable tools to optimize joint positions. Consider gradients of both objectives (distance and material).  
- **Hybrid / Iterative Approaches:**  
  - Cycle between GA and GD to refine solutions.  
  - Compare and save improved populations incrementally (keep solutions only if they improve HV).  
  - Explore more efficient mechanism representations to reduce parameter space.

### Deliverables
For each of the six target curves:  
- Generate and submit up to **1000 valid mechanisms**.  
- Ensure mechanisms are evaluated with the provided tools (`Tools`, `DifferentiableTools`).  
- Organize results so that **hypervolumes can be calculated and compared** against baseline solutions.  

Your goal is to maximize the average hypervolume score across all six problems by producing mechanisms that are both accurate and efficient under the given constraints.

----------------------------------------------------------------------------------------

# Wokflow  
Step 1 – Setup  
Step 2 – Initial Population (feasibility + diversity)  
Step 3 – GA Optimization (seed-based NSGA-II)  
Step 4 – GD Refinement (distance + material, adaptive step)  
Step 5 – Compare, Save, Review. (incremental improvement per problem)  


### Problem Allocation

Berfin: Problem 1, 2, 3  
Sergio: Problem 4, 5, 6

### Round 1 - NODE Explore
Change NODES to 6, 7, 8 to decide best mechanism design for each curve  

Other variables a low compute:  
SEED_PER_SIZE = 100  
TARGET_VALID  = 50   
POP           = 50  
N_GEN         = 50  
GD_STEPS      = 50   
SEEDS = [0, 1]   

| Problem | NODES = 6  | NODES = 7 |NODES = 8 |
|---------|------------|-----------|----------|
| 1       |    2.69    |   4.99    |   1.97   |
| 2       |    2.93    |   0.00    |   0.00   |
| 3       |    3.75    |   4.30    |   2.40   |
| 4       |    3.67    |   2.33    |   2.28   |
| 5       |    0.32    |   1.05    |   1.30   |
| 6       |    1.41    |   0.76    |   1.36   |

### Round 2 - Deep Dive
Fixed NODE to decide best mechanism design for each curve   

Other variables a low compute:  
SEED_PER_SIZE = 3000  
TARGET_VALID  = 200   
POP           = 200  
N_GEN         = 200  
GD_STEPS      = 200   
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 13]  

| Problem |  Hypervolume |  
|---------|--------------| 
| 1       |    5.40      |
| 2       |    3.56      |
| 3       |    4.60      |
| 4       |    5.38      |
| 5       |    3.85      |
| 6       |    3.87      |


# 1. Setup & Global Config

a. Target Optimization Curve + Variables to Change

```python
# --- Configurable target curve ---
Problem_to_solve = 5 # 1-6 # ★★★★★★★★ originally 180 ★★★★★★★★★★


# --- Mechanism size / encoding ---
NODES = 8                       # must match initial pool generator and problem(N=NODES)

# === Seeding (Section 2) ===
SEED_PER_SIZE = 50      # ★★★★★★★★ generate_pool_single_size(..., per_size=SEED_PER_SIZE) // originally 200 ★★★★★★★★★★
TARGET_VALID  = 10      # ★★★★★★★★ stop once this many feasible seeds found // originally 150 ★★★★★★★★★★

# === GA config (Section 3) ===
POP      = 50 # ★★★★★★★★ originally 150 ★★★★★★★★★★
N_GEN    = 50 # ★★★★★★★★ originally 150 ★★★★★★★★★★
MUT_PROB = 0.90 # originally 080
SEEDS = [0, 1] # ★★★★★★★★ originally [0, 1, 2, 3, 4, 5, 6, 7, 11, 13] ★★★★★★★★★★

# Optional GA knobs 
CROSSOVER_PROB = 1.0            # SBX default behavior in pymoo often uses mating config
CROSSOVER_ETA  = 15             # SBX “spread” (bigger => offspring closer to parents)
MUTATION_ETA   = 20             # PM “spread”  (bigger => smaller perturbations)

# === GD refinement (section 4) ===
RUN_GD     = True
GD_TOPK    = 10       # take best-K GA designs into GD
GD_STEPS   = 10      # ★★★★★★★★★★ iterations original 100 ★★★★★★★★★★
GD_STEP    = 3e-4    # base step size
GD_W_DIST  = 0.7     # weight on distance gradient
GD_W_MAT   = 0.3     # weight on material gradient (scaled by /10 in code)
GD_WDIST = 0.7
GD_WMAT  = 0.3
GD_PATIENCE = 10         # patience before decaying step
GD_DECAY    = 0.97      # step decay multiplier


curve_index = Problem_to_solve - 1   # 0-based index: 0 = Problem 1, 1 = Problem 2, ..., 5 = Problem 6
```

b. Environment Setup

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Disable GPU for JAX (Remove if you want to use GPU)

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm, trange

# deteministic random numbers
# np.random.seed(0)
# random.seed(0)

import matplotlib as mpl
plt.style.use('default')  # reset any dark style
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})
```

c. Load Tarject Cruves

```python
target_curves = np.load('target_curves.npy')

# Plot all target curves

# Initialize a 2x3 subplot for plotting all target curves
fig, axs = plt.subplots(2, 3, figsize=(8, 5))

# Loop through the 6 target curves to plot them
for i in range(6):
    # Extract x and y coordinates from the target curve
    x_coords = np.array(target_curves[i])[:, 0]
    y_coords = np.array(target_curves[i])[:, 1]

    # Plot the curve on the respective subplot
    axs[i // 3, i % 3].plot(x_coords, y_coords, color='black', linewidth=3)

    # Set title for each subplot
    axs[i // 3, i % 3].set_title(f'Egg {i + 1}')

    # Ensure equal scaling for x and y axes
    axs[i // 3, i % 3].axis('equal')
    axs[i // 3, i % 3].axis('off')

```


    
![png](output_10_0.png)
    


d. Optimization Imports

```python
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from LINKS.Optimization import DifferentiableTools, Tools

# === Instantiate Tools (non-diff) and DifferentiableTools (diff/grad) ===
try:
    PROBLEM_TOOLS
except NameError:
    from LINKS.Optimization import Tools
    PROBLEM_TOOLS = Tools(device='cpu')
    PROBLEM_TOOLS.compile()

TOOLS = PROBLEM_TOOLS  # make sure GD code can call TOOLS(...)

# Differentiable toolkit for gradients during GD refinement
from LINKS.Optimization import DifferentiableTools
DIFF = DifferentiableTools(device='cpu')
DIFF.compile()

```

e. Problem Definition (Tools + Pymoo class)

```python
PROBLEM_TOOLS = Tools( # define this outside the class due to pymoo deepcopy limitations
            device='cpu' # device to run the optimization on
        )  
PROBLEM_TOOLS.compile() # compile the functions for faster runs

class mechanism_synthesis_optimization(ElementwiseProblem):

    # When intializing, set the mechanism size and target curve
    def __init__(self, target_curve, N = NODES):
        self.N = N
        variables = dict()

        # The upper triangular portion of our NxN Connectivity Matrix consists of Nx(N-1)/2 boolean variables:
        for i in range(N):
            for j in range(i):
                variables["C" + str(j) + "_" + str(i)] = Binary()

        # We Delete C0_1 since we know node 1 is connected to the motor
        del variables["C0_1"]

        #Our position matrix consists of Nx2 real numbers (cartesian coordinate values) between 0 and 1
        for i in range(2*N):
            variables["X0" + str(i)] = Real(bounds=(0.0, 1.0))

        # Our node type vector consists of N boolean variables (fixed vs non-fixed)
        for i in range(N):
            variables["fixed_nodes" + str(i)] =  Binary(N)

        # Our target node is an integer between 1 and N-1, (any except the motor node).
        variables["target"] = Integer(bounds=(1,N-1))

        # Set up some variables in the problem class we inherit for pymoo
        # n_obj=number of objectives, n_constr=number of constraints
        # Our objectives are chamfer distance and material, and they both have constraints.
        super().__init__(vars=variables, n_obj=2, n_constr=2)

        # Store the target curve point cloud
        self.target_curve = target_curve


    def convert_1D_to_mech(self, x):
        N = self.N

        # Get target joints index
        target_idx = x["target"]

        # Build connectivity matrix from its flattened constitutive variables
        C = np.zeros((N,N))
        x["C0_1"] = 1

        for i in range(N):
            for j in range(i):
                # C[i,j] = x["C" + str(j) + "_" + str(i)]
                C[j,i] = x["C" + str(j) + "_" + str(i)]

        edges = np.array(np.where(C==1)).T
        
        # Reshape flattened position matrix to its proper Nx2 shape
        x0 = np.array([x["X0" + str(i)] for i in range(2*N)]).reshape([N,2])

        # Extract a list of Nodes that are fixed from boolean fixed_nodes vector
        fixed_joints = np.where(np.array([x["fixed_nodes" + str(i)] for i in range(N)]))[0].astype(int)

        #We fix the motor and original ground node as 0 and 1 respectively in this implementation
        motor=np.array([0,1])

        return x0, edges, fixed_joints, motor, target_idx

    def convert_mech_to_1D(self, x0, edges, fixed_joints, target_idx=None, **kwargs):
        # This function assumes motor to be [0, 1] our random mechanism generator automatically does this
        N = self.N

        # Initialize dictionary to store 1D representation of mechanism
        x = {}

        # Store target node value
        if target_idx is None:
            target_idx = x0.shape[0]-1 # Assume last node is the target if not specified
            
        x["target"] = target_idx

        # Store connectivity matrix in its flattened form
        C = np.zeros((N,N), dtype=bool)
        C[edges[:,0], edges[:,1]] = 1
        C[edges[:,1], edges[:,0]] = 1
       
        for i in range(N):
            for j in range(i):
                x["C" + str(j) + "_" + str(i)] = C[i,j]

        del x["C0_1"]
        
        # Store position matrix in its flattened form
        if x0.shape[0] != N:
            x0 = np.pad(x0, ((0, N - x0.shape[0]), (0, 0)), 'constant', constant_values=0)
            
        for i in range(2*N):
            x["X0" + str(i)] = x0.flatten()[i]

        # Store fixed nodes in boolean vector form
        for i in range(N):
            x["fixed_nodes" + str(i)] = (i in fixed_joints) or (i>=N)

        return x

    def _evaluate(self, x, out, *args, **kwargs):
        #Convert to mechanism representation
        x0, edges, fixed_joints, motor, target_idx = self.convert_1D_to_mech(x)
        
        # Simulate
        distance, material = PROBLEM_TOOLS(x0,
                                edges,
                                fixed_joints,
                                motor,
                                self.target_curve,
                                target_idx=target_idx
                            )

        out["F"] = np.array([distance, material])
        out["G"] = out["F"] - np.array([0.75, 10.0])  # Constraints: distance <= 0.75, material <= 10.0

```

# 2. Initial Population

a. Imports and Helpers

```python
# === Section 2 — Cell 1: Seeding utilities (single-size, early-stop, ε-dedup) ===
from LINKS.Optimization import MechanismRandomizer, Tools
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm.auto import tqdm
import numpy as np
from time import perf_counter
from collections import defaultdict

# Use the same device everywhere
TOOLS = Tools(device='cpu')
TOOLS.compile()

def epsilon_dedup(F, eps=(5e-4, 5e-3)):
    """
    Keep at most one point per epsilon-box in (distance, material) objective space.
    eps is (eps_distance, eps_material).
    """
    if len(F) == 0:
        return np.array([], dtype=int)
    keys = (np.floor(F[:,0]/eps[0]).astype(int), np.floor(F[:,1]/eps[1]).astype(int))
    seen, keep = set(), []
    for i, key in enumerate(zip(*keys)):
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return np.array(keep, dtype=int)

def generate_pool_single_size(n=7, per_size=SEED_PER_SIZE, desc="Sampling mechanisms"):
    """
    Randomize mechanisms at exactly size n (to match GA encoding N=n).
    """
    t0 = perf_counter()
    pool = []
    randomizer = MechanismRandomizer(min_size=n, max_size=n, device='cpu')
    for _ in tqdm(range(per_size), desc=f"{desc} n={n}", leave=False):
        pool.append(randomizer(n=n))
    print(f"[seed] sampled {len(pool)} candidates at n={n} in {perf_counter()-t0:.1f}s")
    return pool

def evaluate_mechs(mechs, target_curve, problem, diag_every=200,
                   distance_max=0.75, material_max=10.0, max_joints=20,
                   TARGET_VALID=120):
    """
    Evaluate pool, keep only constraint-satisfying candidates, show running diagnostics.
    Early-stops once target_valid feasible candidates are collected.
    """
    t0 = perf_counter()
    X_dicts, F_list, valid = [], [], []
    size_hist = defaultdict(int)
    best = {"distance": np.inf, "material": np.inf}
    ref = np.array([distance_max, material_max])

    pbar = tqdm(total=len(mechs), desc="[eval] checking constraints", leave=False)
    for i, mech in enumerate(mechs, 1):
        x0, edges, fixed_joints, motor = mech['x0'], mech['edges'], mech['fixed_joints'], mech['motor']
        # default: last node is target
        distance, material = TOOLS(x0, edges, fixed_joints, motor, target_curve, target_idx=x0.shape[0]-1)

        best["distance"] = min(best["distance"], distance)
        best["material"] = min(best["material"], material)

        if (distance <= distance_max) and (material <= material_max) and (x0.shape[0] <= max_joints):
            X = problem.convert_mech_to_1D(x0=x0, edges=edges, fixed_joints=fixed_joints,
                                           target_idx=x0.shape[0]-1, motor=motor)
            X_dicts.append(X)
            F_list.append([distance, material])
            valid.append(mech)
            size_hist[x0.shape[0]] += 1

        if (i % diag_every == 0) or (i == len(mechs)):
            acc = len(X_dicts) / i
            pbar.set_postfix({
                "seen": i,
                "valid": len(X_dicts),
                "acc%": f"{100*acc:.1f}",
                "best_d": f"{best['distance']:.3f}",
                "best_m": f"{best['material']:.3f}"
            })
        pbar.update(1)

        # Early stop if we hit target_valid feasible seeds
        if len(X_dicts) >= TARGET_VALID:
            break

    pbar.close()

    F = np.array(F_list) if F_list else np.zeros((0,2))
    print(f"[eval] done in {perf_counter()-t0:.1f}s | valid={len(X_dicts)} "
          f"({100*len(X_dicts)/max(1, min(len(mechs), i)):.1f}%)")
    if len(X_dicts):
        print(f"[eval] valid F ranges: distance [{F[:,0].min():.3f}, {F[:,0].max():.3f}] | "
              f"material [{F[:,1].min():.3f}, {F[:,1].max():.3f}] | ref={tuple(ref)}")
        sizes_sorted = sorted(size_hist.items())
        dist_str = ", ".join([f"n={k}:{v}" for k,v in sizes_sorted])
        print(f"[eval] size distribution among valid: {dist_str}")
    return X_dicts, F, valid

def select_diverse_non_dominated(X_dicts, F, k=100, title="[seed selection]"):
    """
    1) Non-dominated filter, 2) Max-min spread selection to k.
    (Assumes ε-dedup already applied if desired.)
    """
    if len(X_dicts) == 0:
        print(f"{title} no candidates to select."); return []
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    X_nd, F_nd = [X_dicts[i] for i in nds], F[nds]
    print(f"{title} non-dominated front size: {len(X_nd)}")

    if len(X_nd) <= k:
        print(f"{title} selected all {len(X_nd)} (<= k={k})")
        return X_nd

    # Greedy max-min spread
    chosen = [int(np.argmin(F_nd.sum(axis=1)))]
    dist = np.linalg.norm(F_nd - F_nd[chosen[0]], axis=1)
    for _ in tqdm(range(k-1), desc=f"{title} max-min pick", leave=False):
        next_i = int(np.argmax(dist))
        chosen.append(next_i)
        dist = np.minimum(dist, np.linalg.norm(F_nd - F_nd[next_i], axis=1))
    print(f"{title} selected k={k} diverse seeds")
    return [X_nd[i] for i in chosen]

```

b. Generate and Evaluate Pool

```python
# === Section 2 — Cell 2: Build initial pool and filter (single-size matching GA N) ===
GA_N = NODES  # <- single place that ties N to everything downstream

# Always (re)build problem for the current curve & node count so Run All is consistent
problem = mechanism_synthesis_optimization(target_curves[curve_index], N=GA_N)

# Reset seeds so we never reuse a different-N population on Run All
initial_population = []

# Step 1: generate pool (exactly n=GA_N to avoid padding artifacts)
raw_pool = generate_pool_single_size(n=GA_N, per_size=SEED_PER_SIZE)

# Step 2: evaluate and filter with early-stop
X_dicts, F, valid_mechs = evaluate_mechs(
    raw_pool,
    target_curve=target_curves[curve_index],
    problem=problem,
    diag_every=200,
    distance_max=0.75,
    material_max=10.0,
    max_joints=20,
    TARGET_VALID=TARGET_VALID
)

# Use the evaluated dict seeds(never Individuals)
initial_population = X_dicts  # may be empty; GA sampler will handle both cases

```


    Sampling mechanisms n=8:   0%|          | 0/3000 [00:00<?, ?it/s]


    [seed] sampled 3000 candidates at n=8 in 2151.6s
    


    [eval] checking constraints:   0%|          | 0/3000 [00:00<?, ?it/s]


    [eval] done in 15.8s | valid=2 (0.1%)
    [eval] valid F ranges: distance [0.544, 0.690] | material [8.727, 8.984] | ref=(np.float64(0.75), np.float64(10.0))
    [eval] size distribution among valid: n=8:2
    

c. Select Initial Population

```python
# === Section 2 — Cell 3: ε-dedup + pick diverse non-dominated seeds ===
# ε-dedup before selecting spread to reduce near-duplicates
if len(F) > 0:
    dd_idx = epsilon_dedup(F, eps=(5e-4, 5e-3))  # tweak eps if too aggressive
    X_dd = [X_dicts[i] for i in dd_idx]
    F_dd = F[dd_idx]
else:
    X_dd, F_dd = [], F

initial_population = select_diverse_non_dominated(X_dd, F_dd, k=100, title="[seed selection]")
print(f"[seed] → init pop size: {len(initial_population)}")


```

    [seed selection] non-dominated front size: 2
    [seed selection] selected all 2 (<= k=100)
    [seed] → init pop size: 2
    

# 3. GA Optimization

a. GA Optimization Setup

```python
# === GA sampling helper (reads current seeds at call time) ===
from pymoo.core.sampling import Sampling
from pymoo.core.mixed import MixedVariableSampling

class sample_from_seeds(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        n = int(n_samples)
        seeds = initial_population if ('initial_population' in globals() and len(initial_population) > 0) else []
        if len(seeds) > 0:
            # replicate dict seeds to fill the requested pop
            return np.array([seeds[i % len(seeds)] for i in range(n)], dtype=object)
        else:
            # safe fallback: random mixed-variable sampling
            return MixedVariableSampling().do(problem, n).get("X")

```

b. GA Optimization

```python
# === GA run (mixed-variable safe) ===
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

deduper = MixedVariableDuplicateElimination()

algorithm = NSGA2(
    pop_size=POP,
    sampling=sample_from_seeds(),
    mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
    mutation=PolynomialMutation(prob=MUT_PROB),
    eliminate_duplicates=False                                 
)
# Optional sanity check: these must NOT be bool
print("[debug] mating deduper:", type(algorithm.mating.eliminate_duplicates))
print("[debug] algo deduper:  ", type(algorithm.eliminate_duplicates))

results = minimize(
    problem,
    algorithm,
    ('n_gen', N_GEN),
    verbose=True,
    save_history=True,
    seed=int(np.random.randint(1_000_000_000))     # fresh run-to-run seed
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 12
          6 from pymoo.optimize import minimize
          8 deduper = MixedVariableDuplicateElimination()
         10 algorithm = NSGA2(
         11     pop_size=POP,
    ---> 12     sampling=sample_from_seeds(),
         13     mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
         14     mutation=PolynomialMutation(prob=MUT_PROB),
         15     eliminate_duplicates=False                                 
         16 )
         17 # Optional sanity check: these must NOT be bool
         18 print("[debug] mating deduper:", type(algorithm.mating.eliminate_duplicates))
    

    NameError: name 'sample_from_seeds' is not defined


c. Hypervolume Check

```python
from LINKS.Visualization import GAVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine
from pymoo.indicators.hv import HV
ga_visualizer = GAVisualizer()
solver = MechanismSolver(device='cpu')
curve_engine = CurveEngine(device='cpu')

if not results.X is None:
    #Specify reference point
    ref_point = np.array([0.75, 10.0])

    #Calculate Hypervolume
    ind = HV(ref_point)
    hypervolume = ind(results.F)

    #Print and plot
    print('Hyper Volume ~ %f' %(hypervolume))
    ga_visualizer.plot_HV(results.F, ref_point, objective_labels=['Distance', 'Material']) #use the plot_HV function from utils
else:
    print('Did Not Find Solutions!!')
```

    Hyper Volume ~ 2.769152
    


    
![png](output_27_1.png)
    


d. GA multi-seed restarts (union + HV)



```python
# ===  GA Multi-Seed Restarts (union + Pareto + HV) ===
import numpy as np, matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV


SEEDS = [0,1,2,3,4,5] # original [0,1,2,3,4,5,6,7,11,13]
Seeds_List = [0,1,2,3,4,5]

# Config (use existing POP/N_GEN/MUT_PROB; tweak seeds if you like)
POP   = globals().get("POP", 150)
N_GEN = globals().get("N_GEN", 150)
SEEDS = globals().get("SEEDS", Seeds_List)
MUT_PROB = globals().get("MUT_PROB", 0.90)

# Reuse helper; it falls back to random if no seeds are available
assert 'sample_from_seeds' in globals(), "sample_from_seeds() must be defined in 3.b"

# --- Build an algorithm with the safe dedup config ---
def make_algo():
    return NSGA2(
        pop_size=POP,
        sampling=sample_from_seeds(),                                # seeds if present, else random
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination() # dedup at mating (compatible with mixed dict vars)
        ),
        mutation=PolynomialMutation(prob=MUT_PROB),
        eliminate_duplicates=False                                   # turn OFF algorithm-level dedup (prevents 'Individual.items()' crash)
    )

# Helpers to robustly extract final F/X (covers different pymoo versions)
def extract_final_F(res):
    F = getattr(res, "F", None)
    if isinstance(F, np.ndarray) and F.ndim == 2 and len(F):
        return F
    algo = getattr(res, "algorithm", None)
    if algo is not None and hasattr(algo, "pop"):
        Fp = algo.pop.get("F")
        if Fp is not None and len(Fp): return np.asarray(Fp)
    H = getattr(res, "history", None)
    if H:
        e = H[-1]
        try:
            Fh = e.opt.get("F") or e.pop.get("F")
            if Fh is not None and len(Fh): return np.asarray(Fh)
        except Exception:
            pass
    return None

def extract_final_X(res):
    X = getattr(res, "X", None)
    if isinstance(X, np.ndarray) and len(X):
        return X
    algo = getattr(res, "algorithm", None)
    if algo is not None and hasattr(algo, "pop"):
        Xp = algo.pop.get("X")
        if Xp is not None and len(Xp): return np.asarray(Xp, dtype=object)
    H = getattr(res, "history", None)
    if H:
        e = H[-1]
        try:
            Xh = e.opt.get("X") or e.pop.get("X")
            if Xh is not None and len(Xh): return np.asarray(Xh, dtype=object)
        except Exception:
            pass
    return None

def pareto_mask(F):
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(F[i]).all():
            keep[i] = False; continue
        for j in range(n):
            if i == j: continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False; break
    return keep

# --- Run all seeds (no hard failure if initial_population is empty) ---
runs = []
for sd in SEEDS:
    try:
        algo = make_algo()
        res  = minimize(problem, algo, ('n_gen', N_GEN), verbose=False, seed=sd, save_history=True)
        Ffin = extract_final_F(res)
        Xfin = extract_final_X(res)
        if Ffin is None or Xfin is None:
            Ffin = np.empty((0,2)); Xfin = np.empty((0,), dtype=object)
        m = np.isfinite(Ffin).all(axis=1)
        Ffin, Xfin = Ffin[m], Xfin[m]
        runs.append((sd, Ffin, Xfin))
        print(f"[3.e] seed {sd}: {len(Ffin)} pts")
    except Exception as e:
        print(f"[3.e] seed {sd} failed: {e}")
        runs.append((sd, np.empty((0,2)), np.empty((0,), dtype=object)))

# --- Combine & Pareto-prune union ---
allF = np.vstack([F for _,F,_ in runs]) if any(len(F) for _,F,_ in runs) else np.empty((0,2))
allX = np.concatenate([X for _,_,X in runs]) if any(len(X) for _,_,X in runs) else np.empty((0,), dtype=object)

if len(allF):
    mask = pareto_mask(allF)
    UNION_F = allF[mask]
    UNION_X = allX[mask]
else:
    UNION_F = np.empty((0,2)); UNION_X = np.empty((0,), dtype=object)

# --- Plot per-seed + union front ---
plt.figure(figsize=(7,6))
colors = ['tab:blue','tab:green','tab:purple','tab:brown','tab:olive','tab:red']
for k,(sd,F,X) in enumerate(runs):
    if len(F):
        plt.scatter(F[:,0], F[:,1], s=14, alpha=0.35, color=colors[k % len(colors)], label=f"seed {sd}")
if len(UNION_F):
    plt.scatter(UNION_F[:,0], UNION_F[:,1], s=42, edgecolors='k', facecolors='none', linewidths=1.2, label="union Pareto")
    lo = np.quantile(allF, 0.02, axis=0); hi = np.quantile(allF, 0.98, axis=0)
    plt.xlim(lo[0], hi[0]); plt.ylim(lo[1], hi[1])
plt.xlabel("Distance"); plt.ylabel("Material"); plt.title(f"Multi-seed GA (seeds={SEEDS})")
plt.grid(True, alpha=0.3); 
if len(UNION_F): plt.legend()
plt.show()

# --- HV at official reference point ---
if len(UNION_F):
    hv = HV(ref_point=[0.75, 10.0])(UNION_F)
    print(f"Union HV @ ref(0.75,10.0): {hv:.6f}")
else:
    print("No finite points found across seeds.")

# --- Convert union X (dict/object) to mechanism dicts for Step 5 ---
def to_mech_list_from_X(problem, X_array):
    out = []
    for x in X_array:
        x0, e, fj, m, t = problem.convert_1D_to_mech(x)
        out.append({"x0": x0, "edges": e, "fixed_joints": fj, "motor": m, "target_joint": t})
    return out

UNION_MECHS = to_mech_list_from_X(problem, UNION_X) if len(UNION_X) else []
print(f"[multi-seed] union Pareto size: {len(UNION_MECHS)}")

```

    [3.e] seed 0: 22 pts
    [3.e] seed 1: 22 pts
    [3.e] seed 2: 20 pts
    [3.e] seed 3: 26 pts
    [3.e] seed 4: 22 pts
    [3.e] seed 5: 19 pts
    


    
![png](output_30_1.png)
    


    Union HV @ ref(0.75,10.0): 3.746827
    [multi-seed] union Pareto size: 19
    

e. Sanity Check

```python
# === Multi-seed QA / sanity checks ===
import numpy as np
from pymoo.indicators.hv import HV

REF = np.array([0.75, 10.0])
MAX_HV = float(np.prod(REF))  # 7.5

def pareto_mask(F):
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(F[i]).all():
            keep[i] = False; continue
        for j in range(n):
            if i == j: continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False; break
    return keep

# Rebuild ALL from the 'runs' you created in 3.e
allF, sizes = [], []
for sd, F, X in runs:
    allF.append(F)
    sizes.append(len(F))
ALL = np.vstack(allF) if any(sizes) else np.empty((0,2))

mask = pareto_mask(ALL) if len(ALL) else np.array([], dtype=bool)
FRONT = ALL[mask] if len(ALL) else np.empty((0,2))

# Union HV and fraction of theoretical max
hv_union = HV(ref_point=REF)(FRONT) if len(FRONT) else 0.0
print(f"Union HV @ ref {REF.tolist()}: {hv_union:.6f}  ({hv_union/MAX_HV:.1%} of max {MAX_HV:.3f})")
print(f"Union Pareto size: {len(FRONT)}")

# Feasible-only HV (challenge cares about feasible)
if len(FRONT):
    feas = (FRONT[:,0] <= REF[0]) & (FRONT[:,1] <= REF[1])
    FRONT_feas = FRONT[feas]
    hv_feas = HV(ref_point=REF)(FRONT_feas) if len(FRONT_feas) else 0.0
    print(f"Feasible-only HV: {hv_feas:.6f}  ({(hv_feas/MAX_HV if MAX_HV>0 else 0):.1%})")
    print(f"Feasible on union front: {len(FRONT_feas)}/{len(FRONT)}")
else:
    print("No union front points to evaluate feasibility.")

# Per-seed feasible HV and contributions to union front
offsets = np.cumsum([0] + sizes)  # index ranges inside ALL for each seed
contrib = np.zeros(len(runs), dtype=int)
if len(ALL):
    idxs = np.where(mask)[0]
    for idx in idxs:
        k = np.searchsorted(offsets, idx, side='right') - 1
        contrib[k] += 1

print("\nPer-seed stats:")
for i, (sd, F, X) in enumerate(runs):
    if len(F):
        Ff = F[(F[:,0] <= REF[0]) & (F[:,1] <= REF[1])]
        hv_s = HV(ref_point=REF)(Ff) if len(Ff) else 0.0
        print(f"  Seed {sd}: {len(Ff):3d}/{len(F):3d} feasible, HV={hv_s:.6f}, contributes {contrib[i]} union pts")
    else:
        print(f"  Seed {sd}: no points")

# After 3.e builds UNION_MECHS / UNION_F:
if RUN_GD and len(UNION_MECHS):
    # build GD batch from union (top by distance + 0.1*material)
    unionF = np.array([[TOOLS(m["x0"], m["edges"], m["fixed_joints"], m["motor"],
                               target_curves[curve_index], target_idx=m["target_joint"])[0],
                        TOOLS(m["x0"], m["edges"], m["fixed_joints"], m["motor"],
                               target_curves[curve_index], target_idx=m["target_joint"])[1]]
                       for m in UNION_MECHS], float)
    order = np.argsort(unionF[:,0] + 0.1*unionF[:,1])[:min(GD_TOPK, len(UNION_MECHS))]
    GD_BATCH = [UNION_MECHS[i] for i in order]
else:
    GD_BATCH = []


```

    Union HV @ ref [0.75, 10.0]: 3.746827  (50.0% of max 7.500)
    Union Pareto size: 19
    Feasible-only HV: 3.746827  (50.0%)
    Feasible on union front: 19/19
    
    Per-seed stats:
      Seed 0:  22/ 22 feasible, HV=2.762550, contributes 0 union pts
      Seed 1:  22/ 22 feasible, HV=2.891000, contributes 0 union pts
      Seed 2:  20/ 20 feasible, HV=2.737581, contributes 0 union pts
      Seed 3:  26/ 26 feasible, HV=2.624305, contributes 0 union pts
      Seed 4:  22/ 22 feasible, HV=2.577915, contributes 0 union pts
      Seed 5:  19/ 19 feasible, HV=3.746827, contributes 19 union pts
    

# 4. GD Optimization (distance + material aware)

```python
# === 4. GD Refinement (distance + material aware) ===
import numpy as np
from pymoo.indicators.hv import HV
from tqdm.auto import trange

LOG_EVERY   = 5          # how often to print progress

_USE_DIFF_GRADS = None

def _ensure_shape(arr, shape):
    arr = np.asarray(arr, dtype=float)
    if arr.shape == shape:
        return arr
    if arr.size == np.prod(shape):
        return arr.reshape(shape)
    return arr

def _numeric_grads(TOOLS, x, edges_, fj, m, target_curve, ti, h=1e-4):
    """Finite-difference gradients for distance and material."""
    x = np.asarray(x, dtype=float)
    orig_shape = x.shape
    d0, m0 = TOOLS(x, edges_, fj, m, target_curve, target_idx=ti)
    x_flat = x.ravel().copy()
    n = x_flat.size
    gd_flat = np.zeros(n, dtype=float)
    gm_flat = np.zeros(n, dtype=float)
    for k in range(n):
        xk = x_flat.copy()
        hk = h * (1.0 + abs(xk[k]))
        xk[k] += hk
        d1, m1 = TOOLS(xk.reshape(orig_shape), edges_, fj, m, target_curve, target_idx=ti)
        gd_flat[k] = (d1 - d0) / hk
        gm_flat[k] = (m1 - m0) / hk
    return float(d0), float(m0), gd_flat.reshape(orig_shape), gm_flat.reshape(orig_shape)

def _try_diff_raw(DIFF, x, edges_, fj, m, target_curve, ti):
    """Attempt multiple DIFF call signatures."""
    try:
        out = DIFF(x, edges_, fj, m, target_curve, target_idx=ti, with_grads=True)
        if isinstance(out, (tuple, list)) and len(out) >= 4:
            return out[:4]
    except TypeError:
        pass
    try:
        out = DIFF.eval(x, edges_, fj, m, target_curve, target_idx=ti, with_grads=True)
        if isinstance(out, (tuple, list)) and len(out) >= 4:
            return out[:4]
    except (AttributeError, TypeError):
        pass
    for attr in ("gradients", "grads", "evaluate_with_grads"):
        fn = getattr(DIFF, attr, None)
        if callable(fn):
            out = fn(x, edges_, fj, m, target_curve, target_idx=ti)
            if isinstance(out, (tuple, list)) and len(out) >= 4:
                return out[:4]
    out = DIFF(x, edges_, fj, m, target_curve, target_idx=ti)
    if isinstance(out, (tuple, list)) and len(out) >= 4:
        return out[:4]
    raise RuntimeError("DIFF did not return a 4-tuple")

def _get_grads(x, edges_, fj, m, target_curve, ti):
    """Unified gradient getter honoring _USE_DIFF_GRADS."""
    global _USE_DIFF_GRADS
    if _USE_DIFF_GRADS is True:
        try:
            d, mat, gd, gm = _try_diff_raw(DIFF, x, edges_, fj, m, target_curve, ti)
            gd = _ensure_shape(gd, np.asarray(x).shape)
            gm = _ensure_shape(gm, np.asarray(x).shape)
            return float(d), float(mat), gd, gm
        except Exception:
            _USE_DIFF_GRADS = False
            return _numeric_grads(TOOLS, x, edges_, fj, m, target_curve, ti)
    if _USE_DIFF_GRADS is False:
        return _numeric_grads(TOOLS, x, edges_, fj, m, target_curve, ti)
    try:
        d, mat, gd, gm = _try_diff_raw(DIFF, x, edges_, fj, m, target_curve, ti)
        gd = _ensure_shape(gd, np.asarray(x).shape)
        gm = _ensure_shape(gm, np.asarray(x).shape)
        _USE_DIFF_GRADS = True
        print("[GD] Using analytic gradients from DIFF ✓")
        return float(d), float(mat), gd, gm
    except Exception as exc:
        _USE_DIFF_GRADS = False
        print(f"[GD] DIFF gradients unavailable; using numeric finite differences (reason: {exc})")
        return _numeric_grads(TOOLS, x, edges_, fj, m, target_curve, ti)

def differentiable_optimization_tools(x_list, edges_list, fj_list, m_list, target_curve, idx_list):
    d_out, m_out, gd_out, gm_out = [], [], [], []
    for x, edges_, fj, m, ti in zip(x_list, edges_list, fj_list, m_list, idx_list):
        d, mat, gd, gm = _get_grads(x, edges_, fj, m, target_curve, ti)
        d_out.append(d); m_out.append(mat); gd_out.append(gd); gm_out.append(gm)
    return np.asarray(d_out, float), np.asarray(m_out, float), gd_out, gm_out

# ----------------------------
# Run GD Refinement
# ----------------------------
if RUN_GD and 'UNION_MECHS' in globals() and len(UNION_MECHS):
    print("[GD] Building batch from UNION_MECHS…")
    unionF = np.array([[TOOLS(m["x0"], m["edges"], m["fixed_joints"], m["motor"],
                               target_curves[curve_index], target_idx=m["target_joint"])[0],
                        TOOLS(m["x0"], m["edges"], m["fixed_joints"], m["motor"],
                               target_curves[curve_index], target_idx=m["target_joint"])[1]]
                       for m in UNION_MECHS], float)
    # Refine EVERY feasible union point (no K cut, no greedy)
    REF_DMAX, REF_MMAX = 0.75, 10.0  # same thresholds as elsewhere
    feas_idx = [i for i,(d,m) in enumerate(unionF) if np.isfinite(d) and np.isfinite(m)
                and d <= REF_DMAX and m <= REF_MMAX]

    # if nothing is feasible, fall back to all finite
    if not feas_idx:
        feas_idx = [i for i,(d,m) in enumerate(unionF) if np.isfinite(d) and np.isfinite(m)]

    GD_BATCH = [UNION_MECHS[i] for i in feas_idx]



    # Unpack into arrays
    x_cur       = [m["x0"].copy() for m in GD_BATCH]
    edges       = [m["edges"] for m in GD_BATCH]
    fixed_joints= [m["fixed_joints"] for m in GD_BATCH]
    motors      = [m["motor"] for m in GD_BATCH]
    target_idxs = [m["target_joint"] for m in GD_BATCH]

    # Hypervolume helper
    def batch_HV(x_list):
        F = []
        for x, e, fj, mo, ti in zip(x_list, edges, fixed_joints, motors, target_idxs):
            d, mat = TOOLS(x, e, fj, mo, target_curves[curve_index], target_idx=ti)
            F.append([d, mat])
        F = np.array(F, float)
        hv = HV(ref_point=np.array([0.75, 10.0]))(F)
        return hv, F

    hv_best, _ = batch_HV(x_cur)
    x_best = [arr.copy() for arr in x_cur]
    print(f"[GD] Start HV: {hv_best:.6f}  | step={GD_STEP:.2e}  (w_dist={GD_WDIST}, w_mat={GD_WMAT})")

    step = GD_STEP
    no_improve = 0

    for it in trange(GD_STEPS, desc="[GD-lite]"):
        d_arr, m_arr, g_d_list, g_m_list = differentiable_optimization_tools(
            x_cur, edges, fixed_joints, motors, target_curves[curve_index], target_idxs
        )

        dirs = []
        for gd_i, gm_i in zip(g_d_list, g_m_list):
            g = GD_WDIST * gd_i + GD_WMAT * (gm_i / 10.0)

            # guard non-finite
            g = np.where(np.isfinite(g), g, 0.0)

            # safe normalize
            norm = np.linalg.norm(g)
            if not np.isfinite(norm) or norm < 1e-12:
                dirs.append(np.zeros_like(g))   # no move if garbage
            else:
                dirs.append(g / norm)


        # --- Backtracking: only accept a step that doesn't reduce HV ---
        MIN_STEP = 1e-12
        tol = 1e-12
        step_try = step
        accepted = False

        while step_try >= MIN_STEP:
            x_tent = [x_cur[i] - step_try * dirs[i] for i in range(len(x_cur))]
            hv_tent, _ = batch_HV(x_tent)
            if np.isfinite(hv_tent) and hv_tent >= hv_best - tol:
                # accept
                x_cur = x_tent
                step = step_try
                accepted = True
                if hv_tent > hv_best + tol:
                    hv_best = hv_tent
                    x_best = [arr.copy() for arr in x_cur]
                    no_improve = 0
                else:
                    no_improve += 1
                break
            print(f"[GD] rejected (HV drop) → step {step_try:.2e} → {step_try*GD_DECAY:.2e}")
            step_try *= GD_DECAY

        if not accepted:
            # Could not find an acceptable step this iteration
            no_improve += 1
            step *= GD_DECAY


        # occasional pullback toward best (optional but harmless with guard)
        if no_improve and (no_improve % 5 == 0):
            for i in range(len(x_cur)):
                x_cur[i] = 0.5 * x_cur[i] + 0.5 * x_best[i]


        if no_improve >= GD_PATIENCE:
            old_step = step
            step *= GD_DECAY
            no_improve = 0
            print(f"[GD] patience hit → decreasing step: {old_step:.2e} → {step:.2e}")

        hv_now, _ = batch_HV(x_cur)  # log accepted state
        if ((it + 1) % LOG_EVERY == 0) or (it == 0):
            print(f"[GD] it={it+1:4d}/{GD_STEPS}  HV={hv_now:.6f}  best={hv_best:.6f}  step={step:.2e}")


    hv_final, _ = batch_HV(x_best)
    print(f"[GD] Done. Best HV: {hv_best:.6f} | Final HV (best set): {hv_final:.6f} | K={len(x_best)}")

    # Export refined sets
    GD_REFINED_X0S = x_best
    GD_REFINED_MECHS = []
    for Xi, Ei, Fj, Mi, Ti in zip(GD_REFINED_X0S, edges, fixed_joints, motors, target_idxs):
        GD_REFINED_MECHS.append({
            "x0": Xi,
            "edges": Ei,
            "fixed_joints": Fj,
            "motor": Mi,
            "target_joint": int(Ti)
        })
else:
    print("[GD] Skipped (RUN_GD is False or no UNION_MECHS).")

```

    [GD] Building batch from UNION_MECHS…
    [GD] Start HV: 3.746827  | step=1.00e-02  (w_dist=0.7, w_mat=0.3)
    


    [GD-lite]:   0%|          | 0/100 [00:00<?, ?it/s]


    [GD] DIFF gradients unavailable; using numeric finite differences (reason: too many indices for array: array is 1-dimensional, but 2 were indexed)
    [GD] rejected (HV drop) → step 1.00e-02 → 5.00e-03
    [GD] rejected (HV drop) → step 5.00e-03 → 2.50e-03
    [GD] rejected (HV drop) → step 2.50e-03 → 1.25e-03
    [GD] rejected (HV drop) → step 1.25e-03 → 6.25e-04
    [GD] rejected (HV drop) → step 6.25e-04 → 3.13e-04
    [GD] rejected (HV drop) → step 3.13e-04 → 1.56e-04
    [GD] it=   1/100  HV=3.747922  best=3.747922  step=1.56e-04
    [GD] rejected (HV drop) → step 1.56e-04 → 7.81e-05
    [GD] rejected (HV drop) → step 7.81e-05 → 3.91e-05
    [GD] rejected (HV drop) → step 3.91e-05 → 1.95e-05
    [GD] rejected (HV drop) → step 1.95e-05 → 9.77e-06
    [GD] rejected (HV drop) → step 9.77e-06 → 4.88e-06
    [GD] rejected (HV drop) → step 4.88e-06 → 2.44e-06
    [GD] rejected (HV drop) → step 2.44e-06 → 1.22e-06
    [GD] rejected (HV drop) → step 1.22e-06 → 6.10e-07
    [GD] it=   5/100  HV=3.752388  best=3.752388  step=6.10e-07
    [GD] rejected (HV drop) → step 6.10e-07 → 3.05e-07
    [GD] rejected (HV drop) → step 3.05e-07 → 1.53e-07
    [GD] rejected (HV drop) → step 1.53e-07 → 7.63e-08
    [GD] rejected (HV drop) → step 7.63e-08 → 3.81e-08
    [GD] rejected (HV drop) → step 3.81e-08 → 1.91e-08
    [GD] rejected (HV drop) → step 1.91e-08 → 9.54e-09
    [GD] rejected (HV drop) → step 9.54e-09 → 4.77e-09
    [GD] rejected (HV drop) → step 4.77e-09 → 2.38e-09
    [GD] rejected (HV drop) → step 2.38e-09 → 1.19e-09
    [GD] rejected (HV drop) → step 1.19e-09 → 5.96e-10
    [GD] it=  10/100  HV=3.752398  best=3.752398  step=5.96e-10
    [GD] rejected (HV drop) → step 5.96e-10 → 2.98e-10
    [GD] rejected (HV drop) → step 2.98e-10 → 1.49e-10
    [GD] rejected (HV drop) → step 1.49e-10 → 7.45e-11
    [GD] rejected (HV drop) → step 7.45e-11 → 3.73e-11
    [GD] rejected (HV drop) → step 3.73e-11 → 1.86e-11
    [GD] rejected (HV drop) → step 1.86e-11 → 9.31e-12
    [GD] rejected (HV drop) → step 9.31e-12 → 4.66e-12
    [GD] rejected (HV drop) → step 4.66e-12 → 2.33e-12
    [GD] rejected (HV drop) → step 2.33e-12 → 1.16e-12
    [GD] it=  15/100  HV=3.752398  best=3.752398  step=1.16e-12
    [GD] patience hit → decreasing step: 1.16e-12 → 5.82e-13
    [GD] it=  20/100  HV=3.752398  best=3.752398  step=2.91e-13
    [GD] it=  25/100  HV=3.752398  best=3.752398  step=9.09e-15
    [GD] patience hit → decreasing step: 5.68e-16 → 2.84e-16
    [GD] it=  30/100  HV=3.752398  best=3.752398  step=1.42e-16
    [GD] it=  35/100  HV=3.752398  best=3.752398  step=4.44e-18
    [GD] patience hit → decreasing step: 2.78e-19 → 1.39e-19
    [GD] it=  40/100  HV=3.752398  best=3.752398  step=6.94e-20
    [GD] it=  45/100  HV=3.752398  best=3.752398  step=2.17e-21
    [GD] patience hit → decreasing step: 1.36e-22 → 6.78e-23
    [GD] it=  50/100  HV=3.752398  best=3.752398  step=3.39e-23
    [GD] it=  55/100  HV=3.752398  best=3.752398  step=1.06e-24
    [GD] patience hit → decreasing step: 6.62e-26 → 3.31e-26
    [GD] it=  60/100  HV=3.752398  best=3.752398  step=1.65e-26
    [GD] it=  65/100  HV=3.752398  best=3.752398  step=5.17e-28
    [GD] patience hit → decreasing step: 3.23e-29 → 1.62e-29
    [GD] it=  70/100  HV=3.752398  best=3.752398  step=8.08e-30
    [GD] it=  75/100  HV=3.752398  best=3.752398  step=2.52e-31
    [GD] patience hit → decreasing step: 1.58e-32 → 7.89e-33
    [GD] it=  80/100  HV=3.752398  best=3.752398  step=3.94e-33
    [GD] it=  85/100  HV=3.752398  best=3.752398  step=1.23e-34
    [GD] patience hit → decreasing step: 7.70e-36 → 3.85e-36
    [GD] it=  90/100  HV=3.752398  best=3.752398  step=1.93e-36
    [GD] it=  95/100  HV=3.752398  best=3.752398  step=6.02e-38
    [GD] patience hit → decreasing step: 3.76e-39 → 1.88e-39
    [GD] it= 100/100  HV=3.752398  best=3.752398  step=9.40e-40
    [GD] Done. Best HV: 3.752398 | Final HV (best set): 3.752398 | K=19
    

# 5. Save

### 5.a. Compare and Save Improved Solutions (per curve)

```python
# Step 5: sanitize → validate → score → save best-per-problem
import numpy as np, os
from copy import deepcopy
from LINKS.CP import make_empty_submission, evaluate_submission

REF_DMAX, REF_MMAX = 0.75, 10.0
SAVE_PATH = "my_submission_Sergio.npy"   # one file for both read & write

def _to_builtin(obj):
    """Recursively convert numpy types/arrays to pure Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj

def _sanitize_mech(mech):
    """Ensure mechanism dict is JSON-serializable with expected field types."""
    m = deepcopy(mech)
    for k in list(m.keys()):
        m[k] = _to_builtin(m[k])
    # Optional strict casts for indices
    if "motor" in m:
        m["motor"] = [int(v) for v in m["motor"]]
    if "target_joint" in m and not isinstance(m["target_joint"], int):
        m["target_joint"] = int(m["target_joint"])
    return m

def _validate_mechs(cands, target_curve, label="cands", print_first=5):
    """Return indices of feasible candidates and summary stats."""
    ok, bad, stats = [], [], []
    for i, mech in enumerate(cands):
        try:
            x0 = np.asarray(mech["x0"], float)
            e  = np.asarray(mech["edges"], int)
            fj = np.asarray(mech["fixed_joints"], int)
            m  = mech["motor"]
            tj = mech["target_joint"]
            d, mat = TOOLS(x0, e, fj, m, target_curve, target_idx=tj)
            finite = np.isfinite(d) and np.isfinite(mat)
            feasible = finite and (d <= REF_DMAX) and (mat <= REF_MMAX)
            stats.append((i, float(d), float(mat), finite, feasible))
            (ok if feasible else bad).append(i)
        except Exception:
            stats.append((i, np.nan, np.nan, False, False))
            bad.append(i)

    print(f"[step5] {label}: feasible {len(ok)}/{len(cands)} (will send only feasible to scorer)")
    for i, d, mat, finite, feas in stats[:print_first]:
        print(f"  idx={i:3d}  d={d:.4f}  m={mat:.4f}  finite={finite}  feasible={feas}")
    return ok, bad, stats

def _pareto_mask(F):
    """Non-dominated mask for minimization in 2D (distance, material)."""
    F = np.asarray(F, float); n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(F[i]).all():
            keep[i] = False
            continue
        for j in range(n):
            if i == j: 
                continue
            # j dominates i if j <= i elementwise and strictly < in at least one
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    return keep

def _score_set(name, mech_list, key):
    """Build payload with this problem only (cap 1000) and return score."""
    payload = make_empty_submission()
    payload[key] = mech_list[:1000]
    out = evaluate_submission(payload)
    return out["Score Breakdown"].get(key, 0.0)

# ---- Build candidate list from available sources
cand_raw = []
if 'UNION_MECHS' in globals() and len(UNION_MECHS):
    print(f"[step5-prep] Taking {len(UNION_MECHS)} from UNION_MECHS")
    cand_raw.extend(UNION_MECHS)
if 'GD_REFINED_MECHS' in globals() and len(GD_REFINED_MECHS):
    print(f"[step5-prep] Also adding {len(GD_REFINED_MECHS)} GD-refined designs")
    cand_raw.extend(GD_REFINED_MECHS)

assert len(cand_raw) > 0, "candidate_list is empty — nothing to score."

# ---- Sanitize (no structural dedup; you asked to keep everything)
cand_san = [_sanitize_mech(m) for m in cand_raw]
print(f"[step5-prep] After sanitize: {len(cand_san)} candidates")

# ---- Validate against this curve and keep only feasible (scorer ignores infeasible anyway)
ok_idx, bad_idx, stats = _validate_mechs(cand_san, target_curves[curve_index], label="sanitized")
cand_feas = [cand_san[i] for i in ok_idx]
if len(cand_feas) == 0:
    print("[step5] No feasible candidates remain → scorer would return 0. Re-check GA/GD outputs and constraints.")
    # Fall back to finite (even if infeasible) so you can still inspect scoring behavior
    finite_idx = [i for i,(_,d,m,finite,feas) in enumerate(stats) if np.isfinite(d) and np.isfinite(m)]
    cand_feas = [cand_san[i] for i in finite_idx[:1000]]
    print(f"[step5] Falling back to {len(cand_feas)} finite (but not necessarily feasible) for debugging.")

# ---- Score: compare variants and decide
ci = int(curve_index) + 1  # 1-based
key = f"Problem {ci}"
print(f"[step5] Scoring {len(cand_feas)} candidate(s) for {key}")

# Load saved file (same file for read & write)
if os.path.exists(SAVE_PATH):
    saved = np.load(SAVE_PATH, allow_pickle=True).item()
else:
    saved = {}

saved_list = saved.get(key, [])
# Score saved (might be empty)
saved_score = _score_set("saved", saved_list, key) if saved_list else 0.0

# Score candidates-only (cap inside scorer)
cand_score = _score_set("candidates", cand_feas, key)

# Score merged+pruned (saved ∪ candidates)
merged = saved_list + cand_feas
if len(merged):
    # keep feasible only for merge before pruning
    feas_idx, _, _ = _validate_mechs(merged, target_curves[curve_index], label="merged", print_first=0)
    merged = [merged[i] for i in feas_idx]

    # Pareto prune by (distance, material)
    F = []
    for mech in merged:
        x0 = np.asarray(mech["x0"], float)
        e  = np.asarray(mech["edges"], int)
        fj = np.asarray(mech["fixed_joints"], int)
        m  = mech["motor"]
        tj = mech["target_joint"]
        d, mat = TOOLS(x0, e, fj, m, target_curves[curve_index], target_idx=tj)
        F.append([d, mat])
    F = np.asarray(F, float)
    mask = _pareto_mask(F)
    merged = [mech for mech, keep in zip(merged, mask) if keep]
else:
    merged = []

merged_score = _score_set("merged", merged, key) if merged else 0.0

print(f"[step5] Scores for {key} — saved: {saved_score:.6f} | candidates: {cand_score:.6f} | merged: {merged_score:.6f}")

# ---- Pick the best of the three; save only if strictly better than saved
best_list, best_score, src = saved_list, saved_score, "saved"
if cand_score > best_score + 1e-12:
    best_list, best_score, src = cand_feas[:1000], cand_score, "candidates"
if merged_score > best_score + 1e-12:
    best_list, best_score, src = merged[:1000], merged_score, "merged"

if best_score > saved_score + 1e-12:
    print(f"[step5] Choosing {src} (|{len(best_list)}|) with score {best_score:.6f} > saved {saved_score:.6f}")
    saved[key] = best_list
    np.save(SAVE_PATH, saved, allow_pickle=True)
    print(f"[step5] Saved {SAVE_PATH} ✓")
else:
    print("[step5] Not saving; saved set is already best.")


```

    [step5-prep] Taking 19 from UNION_MECHS
    [step5-prep] Also adding 19 GD-refined designs
    [step5-prep] After sanitize: 38 candidates
    [step5] sanitized: feasible 38/38 (will send only feasible to scorer)
      idx=  0  d=0.3054  m=2.8983  finite=True  feasible=True
      idx=  1  d=0.3417  m=2.5384  finite=True  feasible=True
      idx=  2  d=0.2871  m=3.1121  finite=True  feasible=True
      idx=  3  d=0.2718  m=3.1494  finite=True  feasible=True
      idx=  4  d=0.2889  m=2.9365  finite=True  feasible=True
    [step5] Scoring 38 candidate(s) for Problem 5
    [step5] merged: feasible 46/46 (will send only feasible to scorer)
    [step5] Scores for Problem 5 — saved: 3.013802 | candidates: 3.752519 | merged: 3.810430
    [step5] Choosing merged (|27|) with score 3.810430 > saved 3.013802
    [step5] Saved my_submission_Sergio.npy ✓
    

### 5.b. Evaluate Submission Check

```python
import numpy as np
from LINKS.CP import evaluate_submission

# Load submission (dict saved with np.save earlier)
submission = np.load("CP1_submission_SM_BA.npy", allow_pickle=True).item()

# Evaluate and print scores
score = evaluate_submission(submission)
print(score)

```

    {'Overall Score': 4.089393493538676, 'Score Breakdown': {'Problem 1': 5.43752257303457, 'Problem 2': 2.9328630015500465, 'Problem 3': 3.1251822811640793, 'Problem 4': 5.35840210256351, 'Problem 5': 3.810429911956845, 'Problem 6': 3.871961090963005}}
    

### 5.c. Extra: Save File as Markdown

```python
# === Save this notebook as Markdown (same base name) — robust version ===
import os, sys, time, nbformat
from nbconvert import MarkdownExporter

def _detect_notebook_path():
    # 1) Try ipynbname (most reliable when it works)
    try:
        import ipynbname
        return str(ipynbname.path())
    except Exception:
        pass
    # 2) VS Code sometimes exposes this env var; fallback to cwd listing
    for name in os.listdir():
        if name.endswith(".ipynb"):
            return os.path.abspath(name)
    raise RuntimeError("Could not detect .ipynb path. Please save and rerun, or set NB_PATH manually.")

try:
    NB_PATH = _detect_notebook_path()
    BASE, _ = os.path.splitext(NB_PATH)
    MD_PATH = BASE + ".md"
    ASSETS_DIR = BASE + "_files"  # nbconvert's default assets folder naming

    # Load the *saved* notebook file
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Export to Markdown, capturing resources (images, figs)
    md_exporter = MarkdownExporter(exclude_input_prompt=True, exclude_output_prompt=True)
    body, resources = md_exporter.from_notebook_node(nb)

    # Write markdown
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(body)

    # Write assets (if any)
    if resources.get("outputs"):
        os.makedirs(ASSETS_DIR, exist_ok=True)
        for filename, data in resources["outputs"].items():
            with open(os.path.join(ASSETS_DIR, filename), "wb") as f:
                f.write(data)

    # Helpful debug info
    n_cells = sum(1 for _ in nb["cells"])
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(NB_PATH)))
    print("✅ Exported")
    print(f"  Source: {NB_PATH}")
    print(f"  Saved:  {MD_PATH}")
    print(f"  Assets: {ASSETS_DIR} ({len(resources.get('outputs', {}))} files)")
    print(f"  Cells:  {n_cells}")
    print(f"  Note: export reflects on-disk content last saved at {mtime}. Save notebook first.")
except Exception as e:
    print("[export failed]", e)
    print("Tip: Save the notebook, ensure it has a .ipynb filename, then rerun this cell.")

```

    ✅ Exported
      Source: c:\Users\smuti\OneDrive\Desktop\CM_3D-Pen\2155-Optimization-Challenge-Problem\z_Sergio-WIP.ipynb
      Saved:  c:\Users\smuti\OneDrive\Desktop\CM_3D-Pen\2155-Optimization-Challenge-Problem\z_Sergio-WIP.md
      Assets: c:\Users\smuti\OneDrive\Desktop\CM_3D-Pen\2155-Optimization-Challenge-Problem\z_Sergio-WIP_files (3 files)
      Cells:  43
      Note: export reflects on-disk content last saved at 2025-10-03 10:59:55. Save notebook first.
    

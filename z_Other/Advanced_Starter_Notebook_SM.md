---
jupyter:
  kernelspec:
    display_name: .venv
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.9
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# CHALLENGE PROBLEM 1 --- (Sergio) {#challenge-problem-1--sergio}

# 1. SETUP {#1-setup}
:::

::: {.cell .markdown}
a\. Environment Setup
:::

::: {.cell .code execution_count="40"}
``` python
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Disable GPU for JAX (Remove if you want to use GPU)

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm, trange

# deteministic random numbers
np.random.seed(0)
random.seed(0)
```
:::

::: {.cell .markdown}
b\. Load Tarject Cruves
:::

::: {.cell .code execution_count="41"}
``` python
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

::: {.output .display_data}
![](3e97232b9855cfa81ae4ad7aee138cd7da3e50fd.png)
:::
:::

::: {.cell .markdown}
c\. Optimization Imports
:::

::: {.cell .code execution_count="42"}
``` python
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling, Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from LINKS.Optimization import DifferentiableTools, Tools
```
:::

::: {.cell .markdown}
d\. Problem Definition (Tools + Pymoo class)
:::

::: {.cell .code execution_count="43"}
``` python
PROBLEM_TOOLS = Tools( # we have to define this outside the class due to pymoo deepcopy limitations
            device='cpu' # device to run the optimization on
        )  
PROBLEM_TOOLS.compile() # compile the functions for faster runs

class mechanism_synthesis_optimization(ElementwiseProblem):

    # When intializing, set the mechanism size and target curve
    def __init__(self, target_curve, N = 5):
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
:::

::: {.cell .markdown}
# 2. Initial Population {#2-initial-population}
:::

::: {.cell .markdown}
a\. Initial population - Randomized
:::

::: {.cell .code execution_count="44"}
``` python
from LINKS.Optimization import MechanismRandomizer
from LINKS.Visualization import MechanismVisualizer

randomizer = MechanismRandomizer(
    min_size = 6, # smalllest mechanism to sample
    max_size = 14, # largest mechanism to sample
    device='cpu')

visualizer = MechanismVisualizer()

# Now let's generate 100 mechanisms of size 7 and initilize a population for optimization
mechanisms = [randomizer(n=7) for _ in range(100)]
```
:::

::: {.cell .markdown}
# 3. GA Optimization {#3-ga-optimization}

b\. GA Optimization Setup
:::

::: {.cell .code execution_count="45"}
``` python
problem = mechanism_synthesis_optimization(target_curves[1], N=7)

initial_population = [problem.convert_mech_to_1D(**mech) for mech in mechanisms]

class sample_from_random(Sampling):
        def _do(self, problem, n_samples, **kwargs):
                return np.array([initial_population[i%len(initial_population)] for i in range(n_samples)])

F = problem.evaluate(np.array(initial_population))[0]
print(f'Best Distance Performance In random population: {F[:,0].min()}')
print(f'Best Material Performance In random population: {F[:,1].min()}')
```

::: {.output .stream .stdout}
    Best Distance Performance In random population: 1.040202260017395
    Best Material Performance In random population: 5.041465759277344
:::
:::

::: {.cell .markdown}
c\. GA Optimization
:::

::: {.cell .code execution_count="46"}
``` python
algorithm = NSGA2(pop_size=100,
                  sampling=sample_from_random(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  mutation=PolynomialMutation(prob=0.5),
                  eliminate_duplicates=MixedVariableDuplicateElimination())

results = minimize(problem,
                algorithm,
                ('n_gen', 100),
                verbose=True,
                save_history=True,
                seed=123
                )
```

::: {.output .stream .stdout}
    ==========================================================================================
    n_gen  |  n_eval  | n_nds  |     cv_min    |     cv_avg    |      eps      |   indicator  
    ==========================================================================================
         1 |      100 |      1 |  0.9722315073 |  9.1872035480 |             - |             -
         2 |      200 |      1 |  0.9722315073 |  7.6788130569 |             - |             -
         3 |      300 |      1 |  0.9722315073 |  7.0109682727 |             - |             -
         4 |      400 |      1 |  0.9722315073 |  6.0986036992 |             - |             -
         5 |      500 |      1 |  0.9722315073 |  5.4369958925 |             - |             -
         6 |      600 |      1 |  0.9722315073 |  5.1376748872 |             - |             -
         7 |      700 |      1 |  0.9722315073 |  4.7179488993 |             - |             -
         8 |      800 |      1 |  0.9722315073 |  4.7179488993 |             - |             -
         9 |      900 |      1 |  0.9722315073 |  4.1332793140 |             - |             -
        10 |     1000 |      1 |  0.9722315073 |  3.7182028484 |             - |             -
        11 |     1100 |      1 |  0.9722315073 |  3.4813717890 |             - |             -
        12 |     1200 |      1 |  0.9050939083 |  3.3105942798 |             - |             -
        13 |     1300 |      1 |  0.4822773933 |  3.1124474359 |             - |             -
        14 |     1400 |      1 |  0.4822773933 |  3.0443626642 |             - |             -
        15 |     1500 |      1 |  0.4822773933 |  2.9852681637 |             - |             -
        16 |     1600 |      1 |  0.4822773933 |  2.8550715494 |             - |             -
        17 |     1700 |      1 |  0.4822773933 |  2.7562985814 |             - |             -
        18 |     1800 |      1 |  0.4822773933 |  2.7472140324 |             - |             -
        19 |     1900 |      1 |  0.4822773933 |  2.6342141616 |             - |             -
        20 |     2000 |      1 |  0.4822773933 |  2.5810456955 |             - |             -
        21 |     2100 |      1 |  0.4822773933 |  2.5393744695 |             - |             -
        22 |     2200 |      1 |  0.4822773933 |  2.4819366801 |             - |             -
        23 |     2300 |      1 |  0.4822773933 |  2.3865255785 |             - |             -
        24 |     2400 |      1 |  0.4822773933 |  2.3452069497 |             - |             -
        25 |     2500 |      1 |  0.4822773933 |  2.3012774682 |             - |             -
        26 |     2600 |      1 |  0.4822773933 |  2.2923513663 |             - |             -
        27 |     2700 |      1 |  0.4822773933 |  2.2801320660 |             - |             -
        28 |     2800 |      1 |  0.4822773933 |  2.2418632901 |             - |             -
        29 |     2900 |      1 |  0.4822773933 |  2.1565805399 |             - |             -
        30 |     3000 |      1 |  0.4822773933 |  2.1513670790 |             - |             -
        31 |     3100 |      1 |  0.4822773933 |  2.1334295285 |             - |             -
        32 |     3200 |      1 |  0.4822773933 |  2.0762204731 |             - |             -
        33 |     3300 |      1 |  0.4822773933 |  2.0604127133 |             - |             -
        34 |     3400 |      1 |  0.4822773933 |  2.0391860807 |             - |             -
        35 |     3500 |      1 |  0.1319637299 |  1.9923070168 |             - |             -
        36 |     3600 |      1 |  0.1319637299 |  1.9579607272 |             - |             -
        37 |     3700 |      1 |  0.1319637299 |  1.8890623331 |             - |             -
        38 |     3800 |      1 |  0.1319637299 |  1.8597989166 |             - |             -
        39 |     3900 |      1 |  0.1319637299 |  1.8256609631 |             - |             -
        40 |     4000 |      1 |  0.1319637299 |  1.8053529096 |             - |             -
        41 |     4100 |      1 |  0.1319637299 |  1.7509223294 |             - |             -
        42 |     4200 |      1 |  0.1319637299 |  1.7280518138 |             - |             -
        43 |     4300 |      1 |  0.1319637299 |  1.6666859925 |             - |             -
        44 |     4400 |      1 |  0.1319637299 |  1.6282447720 |             - |             -
        45 |     4500 |      1 |  0.1319637299 |  1.6173847055 |             - |             -
        46 |     4600 |      1 |  0.1319637299 |  1.5609451008 |             - |             -
        47 |     4700 |      1 |  0.1319637299 |  1.4451085114 |             - |             -
        48 |     4800 |      1 |  0.1319637299 |  1.4092834759 |             - |             -
        49 |     4900 |      1 |  0.1319637299 |  1.3858145690 |             - |             -
        50 |     5000 |      1 |  0.1319637299 |  1.3397432578 |             - |             -
        51 |     5100 |      1 |  0.1319637299 |  1.2864751959 |             - |             -
        52 |     5200 |      1 |  0.1319637299 |  1.2590730453 |             - |             -
        53 |     5300 |      1 |  0.1319637299 |  1.2386746991 |             - |             -
        54 |     5400 |      1 |  0.1319637299 |  1.1965994656 |             - |             -
        55 |     5500 |      1 |  0.1319637299 |  1.1811319256 |             - |             -
        56 |     5600 |      1 |  0.1319637299 |  1.1649068975 |             - |             -
        57 |     5700 |      1 |  0.1319637299 |  1.1292342925 |             - |             -
        58 |     5800 |      1 |  0.1319637299 |  1.0743226302 |             - |             -
        59 |     5900 |      1 |  0.000000E+00 |  1.0037748969 |             - |             -
        60 |     6000 |      1 |  0.000000E+00 |  0.9850386584 |  0.000000E+00 |             f
        61 |     6100 |      1 |  0.000000E+00 |  0.9384483856 |  0.000000E+00 |             f
        62 |     6200 |      1 |  0.000000E+00 |  0.9125246876 |  0.000000E+00 |             f
        63 |     6300 |      2 |  0.000000E+00 |  0.8868653017 |  1.0000000000 |         ideal
        64 |     6400 |      2 |  0.000000E+00 |  0.8690932077 |  0.000000E+00 |             f
        65 |     6500 |      2 |  0.000000E+00 |  0.8407696921 |  0.000000E+00 |             f
        66 |     6600 |      2 |  0.000000E+00 |  0.8129662460 |  0.000000E+00 |             f
        67 |     6700 |      2 |  0.000000E+00 |  0.7820019442 |  0.000000E+00 |             f
        68 |     6800 |      2 |  0.000000E+00 |  0.7625990862 |  0.000000E+00 |             f
        69 |     6900 |      2 |  0.000000E+00 |  0.7262339526 |  0.000000E+00 |             f
        70 |     7000 |      2 |  0.000000E+00 |  0.7085395747 |  0.000000E+00 |             f
        71 |     7100 |      2 |  0.000000E+00 |  0.6870650905 |  0.000000E+00 |             f
        72 |     7200 |      2 |  0.000000E+00 |  0.6743363386 |  0.000000E+00 |             f
        73 |     7300 |      2 |  0.000000E+00 |  0.6660641545 |  0.000000E+00 |             f
        74 |     7400 |      2 |  0.000000E+00 |  0.6496741754 |  0.000000E+00 |             f
        75 |     7500 |      2 |  0.000000E+00 |  0.6281150764 |  0.000000E+00 |             f
        76 |     7600 |      2 |  0.000000E+00 |  0.6209460992 |  0.000000E+00 |             f
        77 |     7700 |      2 |  0.000000E+00 |  0.5906858414 |  0.000000E+00 |             f
        78 |     7800 |      2 |  0.000000E+00 |  0.5893245703 |  0.000000E+00 |             f
        79 |     7900 |      2 |  0.000000E+00 |  0.5821640283 |  0.000000E+00 |             f
        80 |     8000 |      2 |  0.000000E+00 |  0.5735035640 |  0.000000E+00 |             f
        81 |     8100 |      2 |  0.000000E+00 |  0.5629122776 |  0.000000E+00 |             f
        82 |     8200 |      2 |  0.000000E+00 |  0.5538179916 |  0.000000E+00 |             f
        83 |     8300 |      2 |  0.000000E+00 |  0.5274724948 |  0.000000E+00 |             f
        84 |     8400 |      2 |  0.000000E+00 |  0.5150784492 |  0.000000E+00 |             f
        85 |     8500 |      2 |  0.000000E+00 |  0.5039385486 |  0.000000E+00 |             f
        86 |     8600 |      2 |  0.000000E+00 |  0.4784037626 |  0.000000E+00 |             f
        87 |     8700 |      3 |  0.000000E+00 |  0.4520799059 |  0.0777003865 |         ideal
        88 |     8800 |      3 |  0.000000E+00 |  0.4265001458 |  0.000000E+00 |             f
        89 |     8900 |      3 |  0.000000E+00 |  0.4150849026 |  0.000000E+00 |             f
        90 |     9000 |      3 |  0.000000E+00 |  0.4081350392 |  0.000000E+00 |             f
        91 |     9100 |      3 |  0.000000E+00 |  0.3850902873 |  0.3011943178 |         ideal
        92 |     9200 |      3 |  0.000000E+00 |  0.3703581536 |  0.1387691093 |         ideal
        93 |     9300 |      3 |  0.000000E+00 |  0.3672259688 |  0.000000E+00 |             f
        94 |     9400 |      3 |  0.000000E+00 |  0.3499512911 |  0.000000E+00 |             f
        95 |     9500 |      3 |  0.000000E+00 |  0.3311328912 |  0.2176262365 |             f
        96 |     9600 |      4 |  0.000000E+00 |  0.3188029349 |  0.0859255077 |             f
        97 |     9700 |      5 |  0.000000E+00 |  0.3050780737 |  0.1146682541 |         ideal
        98 |     9800 |      5 |  0.000000E+00 |  0.2868936294 |  0.000000E+00 |             f
        99 |     9900 |      5 |  0.000000E+00 |  0.2774296361 |  0.000000E+00 |             f
       100 |    10000 |      5 |  0.000000E+00 |  0.2630447853 |  0.4666411449 |         ideal
:::
:::

::: {.cell .markdown}
d\. Hypervolume Check
:::

::: {.cell .code execution_count="47"}
``` python
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

::: {.output .stream .stdout}
    Hyper Volume ~ 0.803764
:::

::: {.output .display_data}
![](2b8faa0102c3ef0a6b47a2caa7040413c06c85ab.png)
:::
:::

::: {.cell .markdown}
-   Potential Submission:
:::

::: {.cell .code execution_count="48"}
``` python
from LINKS.CP import make_empty_submission, evaluate_submission

submission = make_empty_submission()

# add out population to submission
if not results.X is None:
    for i in range(results.X.shape[0]):
        if not isinstance(results.X, dict):
            x0_member, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X[i])
        else:
            x0_member, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X)
        
        mech = {
            'x0': x0_member,
            'edges': edges,
            'fixed_joints': fixed_joints,
            'motor': motor,
            'target_joint': target_idx
        }
        
        submission['Problem 2'].append(mech)
```
:::

::: {.cell .markdown}
-   Submission Evaluation
:::

::: {.cell .code execution_count="49"}
``` python
evaluate_submission(submission)
```

::: {.output .execute_result execution_count="49"}
    {'Overall Score': 0.13396070708361663,
     'Score Breakdown': {'Problem 1': 0.0,
      'Problem 2': 0.8037642425016998,
      'Problem 3': 0.0,
      'Problem 4': 0.0,
      'Problem 5': 0.0,
      'Problem 6': 0.0}}
:::
:::

::: {.cell .markdown}
# 4. GD Optimization {#4-gd-optimization}
:::

::: {.cell .markdown}
a\. GD Setup
:::

::: {.cell .code execution_count="63"}
``` python
from LINKS.Optimization import DifferentiableTools

differentiable_optimization_tools = DifferentiableTools(
    device='cpu' # device to run the optimization on
)  
differentiable_optimization_tools.compile() # compile the functions for faster runs

# lets make a batch from our GA results
x0s = []
edges = []
fixed_joints = []
motors = []
target_idxs = []
if not results.X is None:
    if not isinstance(results.X, dict):
        for i in range(results.X.shape[0]):
            x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member  = problem.convert_1D_to_mech(results.X[i])
            x0s.append(x0_member)
            edges.append(edges_member)
            fixed_joints.append(fixed_joints_member)
            motors.append(motor_member)
            target_idxs.append(target_idx_member)
    else:
        x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member  = problem.convert_1D_to_mech(results.X)
        x0s.append(x0_member)
        edges.append(edges_member)
        fixed_joints.append(fixed_joints_member)
        motors.append(motor_member)
        target_idxs.append(target_idx_member)

distances, materials, distance_grads, material_grads = differentiable_optimization_tools(
    x0s,
    edges,
    fixed_joints,
    motors,
    target_curves[1],
    target_idxs
)

# print(materials, distances)
```
:::

::: {.cell .markdown}
GD Optimization - distance-based
:::

::: {.cell .code execution_count="64"}
``` python
x = x0s.copy()

step_size = 4e-4
n_steps = 1000

# keep track of which members are done optimizing
done_optimizing = np.zeros(len(x), dtype=bool)

x_last = x.copy()

for step in trange(n_steps):
    
    # get current distances, materials and gradients
    distances, materials, distance_grads, material_grads = differentiable_optimization_tools(
        x,
        edges,
        fixed_joints,
        motors,
        target_curves[1],
        target_idxs
    )
    
    # only update members that are valid and not done optimizing
    valids = np.where(np.logical_and(distances <= 0.75, materials <= 10.0))[0]
    invalids = np.where(~np.logical_and(distances <= 0.75, materials <= 10.0))[0]
    
    # if a member is invalid, revert to last step and mark as done optimizing
    for i in invalids:
        done_optimizing[i] = True
        x[i] = x_last[i]
    
    # keep a copy of last step
    x_last = x.copy()
    
    # update valid members
    for i in valids:
        if done_optimizing[i]:
            continue
        x[i] = x[i] - step_size * distance_grads[i]
        
    if np.all(done_optimizing):
        print(f'All members are done optimizing at step {step}')
        break
```

::: {.output .display_data}
``` json
{"model_id":"1820f730b15b4afd87b2caeb48fcc2df","version_major":2,"version_minor":0}
```
:::
:::

::: {.cell .markdown}
Combination of populations
:::

::: {.cell .code execution_count="65"}
``` python
combined_x0s = x0s + x
combined_edges = edges + edges
combined_fixed_joints = fixed_joints + fixed_joints
combined_motors = motors + motors
combined_target_idxs = target_idxs + target_idxs
```
:::

::: {.cell .markdown}
Hypervolume Optimization Check
:::

::: {.cell .code execution_count="66"}
``` python
combined_x0s = x0s + x
combined_edges = edges + edges
combined_fixed_joints = fixed_joints + fixed_joints
combined_motors = motors + motors
combined_target_idxs = target_idxs + target_idxs

F = np.array(
    PROBLEM_TOOLS(
        combined_x0s,
        combined_edges,
        combined_fixed_joints,
        combined_motors,
        target_curves[1],
        combined_target_idxs
    )).T

F_before = np.array(
    PROBLEM_TOOLS(
        x0s,
        edges,
        fixed_joints,
        motors,
        target_curves[1],
        target_idxs
    )).T

ref = np.array([0.75, 10.0])
ind = HV(ref)

hv_before = ind(F_before)
hv_after = ind(F)

print(f'Hypervolume before gradient optimization: {hv_before:.4f}, after optimization: {hv_after:.4f}')

plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Before Gradient Optimization')
ga_visualizer.plot_HV(F_before, ref, objective_labels=['Distance', 'Material'], ax=plt.gca())

plt.subplot(1, 2, 2)
plt.title('After Gradient Optimization')
ga_visualizer.plot_HV(F, ref, objective_labels=['Distance', 'Material'], ax=plt.gca())
```

::: {.output .stream .stdout}
    Hypervolume before gradient optimization: 0.8038, after optimization: 1.8311
:::

::: {.output .execute_result execution_count="66"}
    <Axes: title={'center': 'After Gradient Optimization'}, xlabel='Material', ylabel='Distance'>
:::

::: {.output .display_data}
![](f369f0a024ddc07a60557d91b831c94edabe9c8a.png)
:::
:::

::: {.cell .markdown}
A few hints and ideas:

-   Why not do some preprocessing to random mechanisms before running
    GA?
-   Can you use the gradients of both functions in gradient based
    optimization?
-   Can you cycle through multiple optimization runs?
-   Can you mix different kinds of GA? We showed how GA can be used for
    a single mechanism skeleton and how it can be used to also create
    the mechanisms?
-   Can you modify the GA mutations and cross-over?
-   Is there a more efficient representation of mechanisms we can use?
-   Are there smarter gradient-based optimization methods?
:::

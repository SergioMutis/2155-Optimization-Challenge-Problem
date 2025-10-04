# Notebook: Advanced_Starter_Notebook

## Cell 0 — Code (exec_count=1)
```python
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

## Cell 1 — Markdown
# Linkage Synthesis Challenge Problem Advanced Code
Here we will tackle the problem with a more complex GA approach. In this notebook, rather than just optimizing the node positions, we will also optimize the connectivity of the mechanims itself. First let's just load the target curves like we did before:


## Cell 2 — Code (exec_count=2)
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

### Outputs (1)

#### Output 1
```
<Figure size 800x500 with 6 Axes>
```

## Cell 3 — Markdown
## GA Full Mechanism Generation
To set up the problem within Pymoo we will have to come up with a 1D representation of the mechanisms. This code effectively just flattens the matrices and variables describing our mechanism into a single vector. However, this representation scheme is highly suboptimal, and can be improved significantly. As a demonstration, we will slightly improve upon this representation, though we encourage you to take this further. As we discussed, we can replace the edges with a connectivity matrrix. We also know the connectivity matrix is symmetric (if nodes 1 and 2 are connected nodes 2 and 1 are also connected) and has a diagonal equal to zero. To avoid making the optimization come up with the whole connectivity matrix, we will only have it come up with the upper triangular half of the matrix (not including the diagonal). This means the algorithm has to come up with $\frac{N^2-N}{2}$ 0s or 1s (i.e., booleans) to construct the adjacency matrix. We can further simplify the process by selecting a fixed motor position and having the optimization build the rest of the mechanism, removing the need for the optimization to determine the motor as well as reducing the number of 0s and 1s in the upper triangular portion by one (Saving 3 parameters in the representation). To do this we will start with the following mechanism:

<img src="https://i.ibb.co/mGywWgr/start.jpg" alt="start" border="0">

We set Node 0 to ground and Node 1 to be ordinary and the motor to be `[1,0]`. Now we know that the upper triangular portion of the connectivity matrix starts with:

<img src="https://i.ibb.co/5RqKYQN/upper-triangular-fromstart.jpg" alt="upper-triangular-fromstart" border="0">

Since we know the types of the first two nodes, we save another two parameters from the optimization. At this point we will have the same representation as we have before, but with fewer numbers and no motor or N (size of mechanisms) as we generate the mechanism to have a certain size.

<img src="https://i.ibb.co/gv2D0Y5/Optimization-Represenation.png" alt="Optimization-Represenation" border="0">

Now that we have a (slightly) improved 1D representation scheme, we will set up our problem with this many parameters. We also need to specify our constraints! Inequality constraints can be set in Pymoo (See: [constraints in pymoo](https://www.pymoo.org/constraints/index.html)) as less than or equal to zero constraints. We will set the two constraints on distance and material use in our problem.

So based on this lets get started with defining the problem in Pymoo.
<ul>
    <li>
        <code>__init__</code>: We need to differentiate between datatypes because crossover, sampling, and mutation will happen slightly differently based on datatype. As such, we create a dictionary of variables and specify which type of variable each one is.
    </li>
    <li>
<code>convert_1D_to_mech</code>: Reformats our dictionary of variables into design info that our simulator uses (edges, x0, etc.). You will need to update this function if you change the way you parameterize mechanisms. IMPORTANT: This function should not be confused for the. We have also provided a function, <code> convert_mech_to_1D </code> to do the opposite, which you may find useful. It is currently unused and is commented out.
    </li>
    <li>
<code>_evaluate</code>: Next, we define the evaluation function. We construct the mechanism using <code>convert_1D_to_mech</code>. Next, perform the same evaluations as we did in the starter notebook.
    </li>
</ul>

## Cell 4 — Code (exec_count=3)
```python
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

## Cell 5 — Code (exec_count=4)
```python
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

## Cell 6 — Markdown
Now let's try this with NSGA-II and one of the easier target curves. We will initialize with mechanisms with 6 nodes. You can test what happens if you increase this number. You should quickly discover that this naive parameterization does not perform well when initializing the GA with more complex mechanisms:

## Cell 7 — Code (exec_count=5)
```python
problem = mechanism_synthesis_optimization(target_curves[1], N=6)

algorithm = NSGA2(pop_size=100,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                #   mutation=PolynomialMutation(prob=0.9),
                  eliminate_duplicates=MixedVariableDuplicateElimination())

results = minimize(problem,
                algorithm,
                ('n_gen', 100),
                verbose=True,
                save_history=True,
                seed=0
                )
```

### Outputs (1)

#### Output 1
```
==========================================================================================
n_gen  |  n_eval  | n_nds  |     cv_min    |     cv_avg    |      eps      |   indicator  
==========================================================================================
     1 |      100 |      1 |           INF |           INF |             - |             -
     2 |      200 |      1 |  0.7779535055 |           INF |             - |             -
     3 |      300 |      1 |  0.7779535055 |           INF |             - |             -
     4 |      400 |      1 |  0.7779535055 |           INF |             - |             -
     5 |      500 |      1 |  0.7779535055 |           INF |             - |             -
     6 |      600 |      1 |  0.7779535055 |           INF |             - |             -
     7 |      700 |      1 |  0.7779535055 |           INF |             - |             -
     8 |      800 |      1 |  0.7779535055 |           INF |             - |             -
     9 |      900 |      1 |  0.7585495710 |           INF |             - |             -
    10 |     1000 |      1 |  0.7585495710 |           INF |             - |             -
    11 |     1100 |      1 |  0.7585495710 |           INF |             - |             -
    12 |     1200 |      1 |  0.7585495710 |           INF |             - |             -
    13 |     1300 |      1 |  0.7585495710 |           INF |             - |             -
    14 |     1400 |      1 |  0.7585495710 |           INF |             - |             -
    15 |     1500 |      1 |  0.7585495710 |  9.0687639284 |             - |             -
    16 |     1600 |      1 |  0.7585495710 |  4.8196274447 |             - |             -
    17 |     1700 |      1 |  0.7585495710 |  2.6791570091 |             - |             -
    18 |     1800 |      1 |  0.7438501120 |  1.4482443166 |             - |             -
    19 |     1900 |      1 |  0.7426650524 |  0.9804902613 |             - |             -
    20 |     2000 |      1 |  0.7426650524 |  0.8187584710 |             - |             -
    21 |     2100 |      1 |  0.7426650524 |  0.7761739826 |             - |             -
    22 |     2200 |      1 |  0.7419799566 |  0.7611730421 |             - |             -
    23 |     2300 |      1 |  0.7417368889 |  0.7546879148 |             - |             -
    24 |     2400 |      1 |  0.7417368889 |  0.7499447787 |             - |             -
    25 |     2500 |      1 |  0.7417368889 |  0.7472132576 |             - |             -
    26 |     2600 |      1 |  0.7417273521 |  0.7451226294 |             - |             -
    27 |     2700 |      1 |  0.7417273521 |  0.7440936899 |             - |             -
    28 |     2800 |      1 |  0.7417273521 |  0.7431608212 |             - |             -
    29 |     2900 |      1 |  0.7417273521 |  0.7426609862 |             - |             -
    30 |     3000 |      1 |  0.7417271137 |  0.7424374163 |             - |             -
    31 |     3100 |      1 |  0.7417271137 |  0.7422164428 |             - |             -
    32 |     3200 |      1 |  0.7417269945 |  0.7420888221 |             - |             -
    33 |     3300 |      1 |  0.7417269945 |  0.7419577169 |             - |             -
    34 |     3400 |      1 |  0.7417269945 |  0.7418907046 |             - |             -
    35 |     3500 |      1 |  0.7417269945 |  0.7418286502 |             - |             -
    36 |     3600 |      1 |  0.7417269945 |  0.7417928851 |             - |             -
    37 |     3700 |      1 |  0.7417269945 |  0.7417614174 |             - |             -
    38 |     3800 |      1 |  0.7417269945 |  0.7417506611 |             - |             -
    39 |     3900 |      1 |  0.7417269945 |  0.7417443609 |             - |             -
    40 |     4000 |      1 |  0.7417269945 |  0.7417351997 |             - |             -
    41 |     4100 |      1 |  0.7417269945 |  0.7417316175 |             - |             -
    42 |     4200 |      1 |  0.7417268753 |  0.7417296398 |             - |             -
    43 |     4300 |      1 |  0.7417268753 |  0.7417288601 |             - |             -
    44 |     4400 |      1 |  0.7417268753 |  0.7417283630 |             - |             -
    45 |     4500 |      1 |  0.7417266369 |  0.7417277896 |             - |             -
    46 |     4600 |      1 |  0.7417266369 |  0.7417272973 |             - |             -
    47 |     4700 |      1 |  0.7417265177 |  0.7417270958 |             - |             -
    48 |     4800 |      1 |  0.7417265177 |  0.7417270195 |             - |             -
    49 |     4900 |      1 |  0.7417265177 |  0.7417269850 |             - |             -
    50 |     5000 |      1 |  0.7417265177 |  0.7417269444 |             - |             -
    51 |     5100 |      1 |  0.7417265177 |  0.7417268991 |             - |             -
    52 |     5200 |      1 |  0.7417265177 |  0.7417268705 |             - |             -
    53 |     5300 |      1 |  0.7417265177 |  0.7417268515 |             - |             -
    54 |     5400 |      1 |  0.7417265177 |  0.7417268217 |             - |             -
    55 |     5500 |      1 |  0.7417265177 |  0.7417268121 |             - |             -
    56 |     5600 |      1 |  0.7417265177 |  0.7417267847 |             - |             -
    57 |     5700 |      1 |  0.7417265177 |  0.7417267704 |             - |             -
    58 |     5800 |      1 |  0.7417265177 |  0.7417267406 |             - |             -
    59 |     5900 |      1 |  0.7417265177 |  0.7417267108 |             - |             -
    60 |     6000 |      1 |  0.7417265177 |  0.7417267048 |             - |             -
    61 |     6100 |      1 |  0.7417265177 |  0.7417266929 |             - |             -
    62 |     6200 |      1 |  0.7417265177 |  0.7417266762 |             - |             -
    63 |     6300 |      1 |  0.7417265177 |  0.7417266560 |             - |             -
    64 |     6400 |      1 |  0.7417265177 |  0.7417266381 |             - |             -
    65 |     6500 |      1 |  0.7417265177 |  0.7417266202 |             - |             -
    66 |     6600 |      1 |  0.7417265177 |  0.7417266178 |             - |             -
    67 |     6700 |      1 |  0.7417265177 |  0.7417266130 |             - |             -
    68 |     6800 |      1 |  0.7417265177 |  0.7417266095 |             - |             -
    69 |     6900 |      1 |  0.7417265177 |  0.7417266059 |             - |             -
    70 |     7000 |      1 |  0.7417265177 |  0.7417266011 |             - |             -
    71 |     7100 |      1 |  0.7417265177 |  0.7417265952 |             - |             -
    72 |     7200 |      1 |  0.7417265177 |  0.7417265904 |             - |             -
    73 |     7300 |      1 |  0.7417265177 |  0.7417265856 |             - |             -
    74 |     7400 |      1 |  0.7417265177 |  0.7417265773 |             - |             -
    75 |     7500 |      1 |  0.7417265177 |  0.7417265713 |             - |             -
    76 |     7600 |      1 |  0.7417265177 |  0.7417265606 |             - |             -
    77 |     7700 |      1 |  0.7417265177 |  0.7417265511 |             - |             -
    78 |     7800 |      1 |  0.7417265177 |  0.7417265391 |             - |             -
    79 |     7900 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    80 |     8000 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    81 |     8100 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    82 |     8200 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    83 |     8300 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    84 |     8400 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    85 |     8500 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    86 |     8600 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    87 |     8700 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    88 |     8800 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    89 |     8900 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    90 |     9000 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    91 |     9100 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    92 |     9200 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    93 |     9300 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    94 |     9400 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    95 |     9500 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    96 |     9600 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    97 |     9700 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    98 |     9800 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
    99 |     9900 |      1 |  0.7417265177 |  0.7417265177 |             - |             -
   100 |    10000 |      1 |  0.7417265177 |  0.7417265177 |             - |             -

```

## Cell 8 — Code (exec_count=6)
```python
if results.X is None:
    print("No feasible solution found !!!!!")
```

### Outputs (1)

#### Output 1
```
No feasible solution found !!!!!

```

## Cell 9 — Markdown
You can see that if you just run GA you will not be able to find any good solutions, even for smaller mechanisms with 6 joints! 

In the starter notebook we gave you a mechanism to initialize GA with. Now we introduce the `MechanismRandomizer` class which can help you generate random mechanisms of any size you need. Let's see how this class works (The first time you run this it will be a little slow):

## Cell 10 — Code (exec_count=5)
```python
from LINKS.Optimization import MechanismRandomizer
from LINKS.Visualization import MechanismVisualizer
```

## Cell 11 — Code (exec_count=6)
```python
randomizer = MechanismRandomizer(
    min_size = 6, # smalllest mechanism to sample
    max_size = 14, # largest mechanism to sample
    device='cpu')

visualizer = MechanismVisualizer()
```

## Cell 12 — Code (exec_count=9)
```python
random_mechanism = randomizer()

# Let's see what the output looks like (it is exactly like the submission format)
print(random_mechanism)
```

### Outputs (1)

#### Output 1
```
{'x0': array([[0.96048002, 0.37513343],
       [0.99879355, 0.24876935],
       [0.49311482, 0.06676837],
       [0.28653182, 0.63725824],
       [0.8377569 , 0.14618064],
       [0.20960542, 0.46454909]]), 'edges': array([[0, 1],
       [1, 3],
       [2, 3],
       [3, 5],
       [4, 5]]), 'fixed_joints': array([0, 2, 4]), 'motor': array([0, 1])}

```

## Cell 13 — Code (exec_count=10)
```python
# Visualize the random mechanism
visualizer(**random_mechanism)
```

### Outputs (2)

#### Output 1
```
<Axes: >
```

#### Output 2
```
<Figure size 1000x1000 with 1 Axes>
```

## Cell 14 — Code (exec_count=11)
```python
# Now let's generate 100 mechanisms of size 7 and initilize a population for optimization
mechanisms = [randomizer(n=7) for _ in trange(100)]
```

### Outputs (1)

#### Output 1
```
  0%|          | 0/100 [00:00<?, ?it/s]
```

## Cell 15 — Code (exec_count=12)
```python
problem = mechanism_synthesis_optimization(target_curves[1], N=7)

initial_population = [problem.convert_mech_to_1D(**mech) for mech in mechanisms]

class sample_from_random(Sampling):
        def _do(self, problem, n_samples, **kwargs):
                return np.array([initial_population[i%len(initial_population)] for i in range(n_samples)])

F = problem.evaluate(np.array(initial_population))[0]
print(f'Best Distance Performance In random population: {F[:,0].min()}')
print(f'Best Material Performance In random population: {F[:,1].min()}')
```

### Outputs (1)

#### Output 1
```
Best Distance Performance In random population: 1.4918473958969116
Best Material Performance In random population: 3.9050841331481934

```

## Cell 16 — Code (exec_count=13)
```python
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

### Outputs (1)

#### Output 1
```
==========================================================================================
n_gen  |  n_eval  | n_nds  |     cv_min    |     cv_avg    |      eps      |   indicator  
==========================================================================================
     1 |      100 |      1 |  0.8068205118 |  9.0035807598 |             - |             -
     2 |      200 |      1 |  0.8068205118 |  7.9413599074 |             - |             -
     3 |      300 |      1 |  0.3041574955 |  6.7534441745 |             - |             -
     4 |      400 |      1 |  0.3041574955 |  5.9106362784 |             - |             -
     5 |      500 |      1 |  0.3041574955 |  5.6705343354 |             - |             -
     6 |      600 |      1 |  0.3041574955 |  5.2590543783 |             - |             -
     7 |      700 |      1 |  0.1190508008 |  4.3487125355 |             - |             -
     8 |      800 |      1 |  0.1190508008 |  4.0359340101 |             - |             -
     9 |      900 |      1 |  0.1190508008 |  3.8640289956 |             - |             -
    10 |     1000 |      1 |  0.1190508008 |  3.6347331434 |             - |             -
    11 |     1100 |      1 |  0.1190508008 |  3.3509976017 |             - |             -
    12 |     1200 |      1 |  0.1190508008 |  3.2090770793 |             - |             -
    13 |     1300 |      1 |  0.1190508008 |  3.1698704553 |             - |             -
    14 |     1400 |      1 |  0.1190508008 |  3.0684441257 |             - |             -
    15 |     1500 |      1 |  0.1190508008 |  3.0447802639 |             - |             -
    16 |     1600 |      1 |  0.1190508008 |  2.9564629877 |             - |             -
    17 |     1700 |      1 |  0.1190508008 |  2.9069718468 |             - |             -
    18 |     1800 |      1 |  0.1190508008 |  2.8143695247 |             - |             -
    19 |     1900 |      1 |  0.000000E+00 |  2.6706756020 |             - |             -
    20 |     2000 |      1 |  0.000000E+00 |  2.5993937087 |  0.000000E+00 |             f
    21 |     2100 |      1 |  0.000000E+00 |  2.4986077332 |  0.000000E+00 |             f
    22 |     2200 |      1 |  0.000000E+00 |  2.4179336548 |  0.000000E+00 |             f
    23 |     2300 |      1 |  0.000000E+00 |  2.3823533368 |  0.000000E+00 |             f
    24 |     2400 |      1 |  0.000000E+00 |  2.3017100984 |  0.000000E+00 |             f
    25 |     2500 |      2 |  0.000000E+00 |  2.1228608423 |  1.0000000000 |         ideal
    26 |     2600 |      2 |  0.000000E+00 |  2.0418268609 |  0.000000E+00 |             f
    27 |     2700 |      2 |  0.000000E+00 |  2.0202888536 |  0.000000E+00 |             f
    28 |     2800 |      2 |  0.000000E+00 |  1.9851009178 |  0.000000E+00 |             f
    29 |     2900 |      2 |  0.000000E+00 |  1.8957750171 |  0.000000E+00 |             f
    30 |     3000 |      2 |  0.000000E+00 |  1.8500430399 |  0.000000E+00 |             f
    31 |     3100 |      2 |  0.000000E+00 |  1.7927172250 |  0.000000E+00 |             f
    32 |     3200 |      2 |  0.000000E+00 |  1.7764744943 |  0.000000E+00 |             f
    33 |     3300 |      2 |  0.000000E+00 |  1.7418416375 |  0.000000E+00 |             f
    34 |     3400 |      2 |  0.000000E+00 |  1.6992622846 |  0.000000E+00 |             f
    35 |     3500 |      3 |  0.000000E+00 |  1.6611794800 |  0.7393613791 |         ideal
    36 |     3600 |      3 |  0.000000E+00 |  1.5868125504 |  0.000000E+00 |             f
    37 |     3700 |      3 |  0.000000E+00 |  1.5447637171 |  0.000000E+00 |             f
    38 |     3800 |      3 |  0.000000E+00 |  1.5161452323 |  0.000000E+00 |             f
    39 |     3900 |      3 |  0.000000E+00 |  1.4683877653 |  0.000000E+00 |             f
    40 |     4000 |      3 |  0.000000E+00 |  1.4115080762 |  0.000000E+00 |             f
    41 |     4100 |      3 |  0.000000E+00 |  1.3657098591 |  0.000000E+00 |             f
    42 |     4200 |      3 |  0.000000E+00 |  1.3400679195 |  0.000000E+00 |             f
    43 |     4300 |      3 |  0.000000E+00 |  1.3008276236 |  0.000000E+00 |             f
    44 |     4400 |      3 |  0.000000E+00 |  1.2502418256 |  0.000000E+00 |             f
    45 |     4500 |      4 |  0.000000E+00 |  1.1832850164 |  0.0678717192 |             f
    46 |     4600 |      4 |  0.000000E+00 |  1.1493733269 |  0.000000E+00 |             f
    47 |     4700 |      4 |  0.000000E+00 |  1.0940864342 |  0.000000E+00 |             f
    48 |     4800 |      4 |  0.000000E+00 |  1.0582112044 |  0.000000E+00 |             f
    49 |     4900 |      4 |  0.000000E+00 |  1.0369527155 |  0.000000E+00 |             f
    50 |     5000 |      5 |  0.000000E+00 |  0.9787474388 |  0.0180666437 |             f
    51 |     5100 |      5 |  0.000000E+00 |  0.9666320056 |  0.000000E+00 |             f
    52 |     5200 |      5 |  0.000000E+00 |  0.9350901788 |  0.000000E+00 |             f
    53 |     5300 |      5 |  0.000000E+00 |  0.8822630477 |  0.000000E+00 |             f
    54 |     5400 |      5 |  0.000000E+00 |  0.8400440264 |  0.000000E+00 |             f
    55 |     5500 |      5 |  0.000000E+00 |  0.8088058984 |  0.000000E+00 |             f
    56 |     5600 |      5 |  0.000000E+00 |  0.7857882631 |  0.000000E+00 |             f
    57 |     5700 |      5 |  0.000000E+00 |  0.7629309678 |  0.000000E+00 |             f
    58 |     5800 |      5 |  0.000000E+00 |  0.7281596404 |  0.000000E+00 |             f
    59 |     5900 |      5 |  0.000000E+00 |  0.6949615914 |  0.000000E+00 |             f
    60 |     6000 |      3 |  0.000000E+00 |  0.6660880369 |  0.0203818144 |             f
    61 |     6100 |      3 |  0.000000E+00 |  0.6233269781 |  0.000000E+00 |             f
    62 |     6200 |      3 |  0.000000E+00 |  0.6070313126 |  0.000000E+00 |             f
    63 |     6300 |      3 |  0.000000E+00 |  0.5611735791 |  0.000000E+00 |             f
    64 |     6400 |      3 |  0.000000E+00 |  0.5356497329 |  0.000000E+00 |             f
    65 |     6500 |      3 |  0.000000E+00 |  0.5172069478 |  0.000000E+00 |             f
    66 |     6600 |      3 |  0.000000E+00 |  0.5007765269 |  0.000000E+00 |             f
    67 |     6700 |      3 |  0.000000E+00 |  0.4988419211 |  0.000000E+00 |             f
    68 |     6800 |      3 |  0.000000E+00 |  0.4909682512 |  0.000000E+00 |             f
    69 |     6900 |      3 |  0.000000E+00 |  0.4840079880 |  0.000000E+00 |             f
    70 |     7000 |      3 |  0.000000E+00 |  0.4752098256 |  0.000000E+00 |             f
    71 |     7100 |      3 |  0.000000E+00 |  0.4676798075 |  0.000000E+00 |             f
    72 |     7200 |      3 |  0.000000E+00 |  0.4491393334 |  0.000000E+00 |             f
    73 |     7300 |      3 |  0.000000E+00 |  0.4368980396 |  0.000000E+00 |             f
    74 |     7400 |      3 |  0.000000E+00 |  0.4218685651 |  0.000000E+00 |             f
    75 |     7500 |      3 |  0.000000E+00 |  0.3965374798 |  0.1691785092 |         ideal
    76 |     7600 |      3 |  0.000000E+00 |  0.3701842898 |  0.000000E+00 |             f
    77 |     7700 |      4 |  0.000000E+00 |  0.3498155934 |  0.0858789075 |             f
    78 |     7800 |      4 |  0.000000E+00 |  0.3230846983 |  0.0075345806 |         ideal
    79 |     7900 |      4 |  0.000000E+00 |  0.2933507830 |  0.0225581258 |             f
    80 |     8000 |      4 |  0.000000E+00 |  0.2818084139 |  0.000000E+00 |             f
    81 |     8100 |      4 |  0.000000E+00 |  0.2274951202 |  0.000000E+00 |             f
    82 |     8200 |      4 |  0.000000E+00 |  0.1885172659 |  0.000000E+00 |             f
    83 |     8300 |      4 |  0.000000E+00 |  0.1530350846 |  0.000000E+00 |             f
    84 |     8400 |      4 |  0.000000E+00 |  0.1258841366 |  0.000000E+00 |             f
    85 |     8500 |      2 |  0.000000E+00 |  0.1182612425 |  4.5026210175 |         nadir
    86 |     8600 |      2 |  0.000000E+00 |  0.0995244396 |  0.000000E+00 |             f
    87 |     8700 |      2 |  0.000000E+00 |  0.0866257304 |  0.000000E+00 |             f
    88 |     8800 |      2 |  0.000000E+00 |  0.0598720539 |  0.000000E+00 |             f
    89 |     8900 |      2 |  0.000000E+00 |  0.0502843219 |  0.000000E+00 |             f
    90 |     9000 |      3 |  0.000000E+00 |  0.0400978184 |  0.1718572640 |             f
    91 |     9100 |      2 |  0.000000E+00 |  0.0227173376 |  0.5943257818 |         ideal
    92 |     9200 |      2 |  0.000000E+00 |  0.0142814791 |  0.000000E+00 |             f
    93 |     9300 |      2 |  0.000000E+00 |  0.0096937597 |  0.000000E+00 |             f
    94 |     9400 |      3 |  0.000000E+00 |  0.0010715401 |  0.2031345078 |             f
    95 |     9500 |      3 |  0.000000E+00 |  0.000000E+00 |  0.000000E+00 |             f
    96 |     9600 |      3 |  0.000000E+00 |  0.000000E+00 |  0.000000E+00 |             f
    97 |     9700 |      3 |  0.000000E+00 |  0.000000E+00 |  0.000000E+00 |             f
    98 |     9800 |      3 |  0.000000E+00 |  0.000000E+00 |  0.000000E+00 |             f
    99 |     9900 |      4 |  0.000000E+00 |  0.000000E+00 |  0.3430985643 |         ideal
   100 |    10000 |      4 |  0.000000E+00 |  0.000000E+00 |  0.000000E+00 |             f

```

## Cell 17 — Markdown
We see that if we initialize GA with some valid samples, it is much more likely to succeed! Let's take a look at the solutions of this run:

## Cell 18 — Code (exec_count=8)
```python
from LINKS.Visualization import GAVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine
```

## Cell 19 — Code (exec_count=15)
```python
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

### Outputs (2)

#### Output 1
```
Hyper Volume ~ 0.786230

```

#### Output 2
```
<Figure size 600x600 with 1 Axes>
```

## Cell 20 — Markdown
Now Let's visualize the best solution for distance

## Cell 21 — Code (exec_count=16)
```python
if not results.X is None:
    if type(results.X)!=dict:
        best_idx = np.argmin(results.F[:,0])
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X[best_idx])
    else:
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X)
else:
    print('Did Not Find Solutions!!')
```

## Cell 22 — Code (exec_count=17)
```python
plt.figure(figsize=(8,8))
visualizer(best_sol, edges, fixed_joints, motor, highlight=target_idx, ax=plt.gca())

traced_curve = solver(best_sol, edges, fixed_joints, motor)[target_idx]
curve_engine.visualize_comparison(
    traced_curve,
    target_curves[1]
)

print(f'Distance: {results.F[best_idx,0]:.4f}, Material: {results.F[best_idx,1]:.4f}')
```

### Outputs (3)

#### Output 1
```
Distance: 0.4240, Material: 8.1902

```

#### Output 2
```
<Figure size 800x800 with 1 Axes>
```

#### Output 3
```
<Figure size 400x400 with 1 Axes>
```

## Cell 23 — Markdown
Now Let's visualize the best solution for material

## Cell 24 — Code (exec_count=18)
```python
if not results.X is None:
    if type(results.X)!=dict:
        best_idx = np.argmin(results.F[:,1])
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X[best_idx])
    else:
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X)
else:
    print('Did Not Find Solutions!!')
```

## Cell 25 — Code (exec_count=19)
```python
plt.figure(figsize=(8,8))
visualizer(best_sol, edges, fixed_joints, motor, ax=plt.gca())

traced_curve = solver(best_sol, edges, fixed_joints, motor)[target_idx]
curve_engine.visualize_comparison(
    traced_curve,
    target_curves[1]
)

print(f'Distance: {results.F[best_idx,0]:.4f}, Material: {results.F[best_idx,1]:.4f}')
```

### Outputs (3)

#### Output 1
```
Distance: 0.5606, Material: 7.3921

```

#### Output 2
```
<Figure size 800x800 with 1 Axes>
```

#### Output 3
```
<Figure size 400x400 with 1 Axes>
```

## Cell 26 — Markdown
You can use this population to make a submission:

## Cell 27 — Code (exec_count=20)
```python
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

## Cell 28 — Code (exec_count=21)
```python
evaluate_submission(submission)
```

### Outputs (1)

#### Output 1
```
{'Overall Score': 0.13103841360318333,
 'Score Breakdown': {'Problem 1': 0.0,
  'Problem 2': 0.7862304816190999,
  'Problem 3': 0.0,
  'Problem 4': 0.0,
  'Problem 5': 0.0,
  'Problem 6': 0.0}}
```

## Cell 29 — Markdown
You can see that GA is still not doing great with handling different mechanisms. You could adjust the initialization more, change mutation and cross-over functionality. So far however, we have not yet touched on gradient-based optimization. What if we took the solutions of GA and optimized for each of the objectives.

Now let's take the GA solutions from above and just optimize for the lowest distance. Then we will add the optimized solutions to the population we have and see how the hypervolume changes.

 **Note that if the material use exceeds 10 during this optimization the score will not change How could you address this?** 

## Cell 30 — Code (exec_count=22)
```python
from LINKS.Optimization import DifferentiableTools
```

## Cell 31 — Code (exec_count=23)
```python
differentiable_optimization_tools = DifferentiableTools(
    device='cpu' # device to run the optimization on
)  
differentiable_optimization_tools.compile() # compile the functions for faster runs
```

## Cell 32 — Code (exec_count=24)
```python
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
```

## Cell 33 — Markdown
here is how this differentiable class works:

## Cell 34 — Code (exec_count=25)
```python
distances, materials, distance_grads, material_grads = differentiable_optimization_tools(
    x0s,
    edges,
    fixed_joints,
    motors,
    target_curves[1],
    target_idxs
)

print(materials, distances)
```

### Outputs (1)

#### Output 1
```
[8.190159  7.6659107 7.983989  7.3920965] [0.42399657 0.4758202  0.47286645 0.5605692 ]

```

## Cell 35 — Markdown
**NOTE:** Sometimes the mechanisms are close to locking, which results in gradients exploding and in these cases the `DifferentiableTools` will return inifity results. You can ignore these since the `Tools` class will still simulate them eventhough the gradients are near singular.

## Cell 36 — Code (exec_count=26)
```python
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

### Outputs (1)

#### Output 1
```
  0%|          | 0/1000 [00:00<?, ?it/s]
```

## Cell 37 — Markdown
Now we can combine these new memebers with the old and see how the hypervolume changes:

## Cell 38 — Code (exec_count=27)
```python
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
```

### Outputs (1)

#### Output 1
```
Hypervolume before gradient optimization: 0.7862, after optimization: 1.0004

```

## Cell 39 — Code (exec_count=28)
```python
plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Before Gradient Optimization')
ga_visualizer.plot_HV(F_before, ref, objective_labels=['Distance', 'Material'], ax=plt.gca())

plt.subplot(1, 2, 2)
plt.title('After Gradient Optimization')
ga_visualizer.plot_HV(F, ref, objective_labels=['Distance', 'Material'], ax=plt.gca())
```

### Outputs (2)

#### Output 1
```
<Axes: title={'center': 'After Gradient Optimization'}, xlabel='Material', ylabel='Distance'>
```

#### Output 2
```
<Figure size 1200x600 with 2 Axes>
```

## Cell 40 — Markdown
A small improvement. But this is just to give you an idea of what you can do when mixing optimization methods.

Now you should have a good idea of how to use the tools we have provided to you. Think about how you can improve each aspect of the optimization to achieve a higher score in you final submission. You are allowed also submit your teams best solutions to the leaderboard which we will announce soon and see how well you are doing compared other teams.

## Cell 41 — Markdown
A few hints and ideas:
- Why not do some preprocessing to random mechanisms before running GA?
- Can you use the gradients of both functions in gradient based optimization? 
- Can you cycle through multiple optimization runs?
- Can you mix different kinds of GA? We showed how GA can be used for a single mechanism skeleton and how it can be used to also create the mechanisms?
- Can you modify the GA mutations and cross-over?
- Is there a more efficient representation of mechanisms we can use?
- Are there smarter gradient-based optimization methods?

## Cell 42 — Markdown
# Appendix and Details

## Curve Distance Calculation

The target curves are provided to you as a sequence of points. The curves your mechanisms generate are then re-oriented compared to the target curves as we showed above. We measure the closeness of two point curves by computing the distance between the two curves with optimal alignment. This distance can be mathematically represented as:

$$
d_{OD}=\frac{2 \pi}{N}\min_{o_1 \in O_1, \alpha \in [0,2\pi)} \sum_{i=1}^N\left\|N(\mathbf{X}^{\text{coupler}}_{o_1(i)})\mathbf{R}(\alpha)-N(\mathbf{X}^{\text{target}})\right\|^2,
$$

$$\operatorname{N}(\mathbf{X})=\frac{\mathbf{X}- \overline{\mathbf{X}}}{\sqrt{\frac{\sum_{i=1}^N\|\mathbf{X}_i-\overline{\mathbf{X}}\|_2^2}{N}}}$$

Where $\mathbf{X}^{\text{coupler}}$ is the curve traced by the mechanism and $\mathbf{X}^{\text{target}}$ is the target curve and  $O_1$ is the set of all possible clockwise and counterclockwise orderings of a curve with $N$ ordered points and $\mathbf{R}(\alpha)$ is a $2\times2$ rotation matrix for angle $\alpha$, while $o_1(i)$ refers to the index of the $i$-th point in the set $o_1$. In practice, in our problem we may ignore the denominator in $N$ since we do not want to optimally rescale curves. The code we provide solves this optimization problem and retuns the distance value so you don't have to worry about this.

The other objective in this challenge problem is reducing the material used by the mechanism. This is the total length of all links in the mechanism. We have provided a function that makes this measurement for you using initial positions and adjacency matrix for a mechanism (The functions and useful tools are going to be described later).

## Cell 43 — Markdown
### A Note On Scale Normalization

## Cell 44 — Markdown
The functions that we saw in the starter notebool on curves, namely `compare_curves` and `visualize_comparison`, and `visualize_alignment` have a scaling option. An important input to the `CurveEngine` class is `normalize_scale`, which if set to `True` will rescale curves when comparing. In our problem we want this to be `False` since we care about the scale and material used in the problem. (We stress this distinction because the first component of the optimization demo had this rescaling turned on!)

## Cell 45 — Code (exec_count=9)
```python
curve_processor = CurveEngine(
    normalize_scale=True, # whether to normalize the curve's scale (This would mean we do not care about the size of the curve but we do!)
    device='cpu' # device to run the processing on
)

curve_processor_unscaled = CurveEngine(
    normalize_scale=False, # do not normalize the scale of the curve (This would mean we care about the size of the curve too!)
    device='cpu'
)
```

## Cell 46 — Markdown
Let's go back to the simple fourbar we had and see what this means:

## Cell 47 — Code (exec_count=10)
```python
visualizer = MechanismVisualizer()

# Define the variables for the mechanism
x0 = np.array([[0.3,0.2],
               [0.6,0.2],
               [0.3,0.3],
               [0.6,0.4],
               [0.4,0.5]])

# Define the connectivity of the mechanism
edges = np.array([[0,2],
                  [1,3],
                  [2,3],
                  [2,4],
                  [3,4]])

# Define the fixed nodes
fixed_joints = np.array([0,1])

# actuator
motor = np.array([0,2])

plt.figure(figsize=(5,5))
visualizer(x0, edges, fixed_joints, motor, ax=plt.gca())
```

### Outputs (2)

#### Output 1
```
<Axes: >
```

#### Output 2
```
<Figure size 500x500 with 1 Axes>
```

## Cell 48 — Code (exec_count=12)
```python
solver = MechanismSolver(
    device = 'cpu' # device to run the simulation on (cpu or gpu)
)

solution = solver(x0, edges, fixed_joints, motor)
traced_curve = solution[4]
```

## Cell 49 — Code (exec_count=13)
```python
# lets compare to the first target curve
distance = curve_processor.compare_curves(traced_curve, target_curves[0])
unscaled_distance = curve_processor_unscaled.compare_curves(traced_curve, target_curves[0])

print(f"The scale normalized distance to target curve 1 is: {distance:.4f}")
print(f"The unnormalized distance to target curve 1 is: {unscaled_distance:.4f}") # This will be higher since do not normalize scale
```

### Outputs (1)

#### Output 1
```
The scale normalized distance to target curve 1 is: 1.4590
The unnormalized distance to target curve 1 is: 3.3600

```

## Cell 50 — Markdown
Now let's visualize the alignment process with and without scale to understand what is going on:

## Cell 51 — Code (exec_count=14)
```python
curve_processor.visualize_alignment(traced_curve, target_curves[0])
curve_processor.visualize_comparison(traced_curve, target_curves[0])
```

### Outputs (2)

#### Output 1
```
<Figure size 800x400 with 2 Axes>
```

#### Output 2
```
<Figure size 400x400 with 1 Axes>
```

## Cell 52 — Code (exec_count=15)
```python
curve_processor_unscaled.visualize_alignment(traced_curve, target_curves[0])
curve_processor_unscaled.visualize_comparison(traced_curve, target_curves[0])
```

### Outputs (2)

#### Output 1
```
<Figure size 800x400 with 2 Axes>
```

#### Output 2
```
<Figure size 400x400 with 1 Axes>
```

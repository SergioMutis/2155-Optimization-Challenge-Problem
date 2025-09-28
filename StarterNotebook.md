# Notebook: StarterNotebook

## Cell 0 — Markdown
# Sunny Side Sketch Challenge Problem

<img src="https://i.ibb.co/MknQ67tW/eggs.png" width="100%" style="max-width: 600px; display:block; margin:auto;">


Once upon a time, an ambitious engineer was tasked with inventing an “egg-maker” — a marvelous machine that could whip up eggs in all their delicious forms: fried, scrambled, poached, you name it. But somewhere along the way, wires got crossed (literally), and instead of producing eggs, the machine started… drawing them. Perfectly neat outlines of every egg recipe you could imagine appeared on paper, but not a single breakfast was served.
And so, the legend of the “Egg-Maker-That-Only-Makes-Egg-DRAWINGS” was born.

Your task: Design 2D planar linkage mechanisms that can trace six target egg-paths. These curves are your recipes — now go cook up some mechanisms!



## Cell 1 — Code (exec_count=1)
```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Disable GPU for JAX (Remove if you want to use GPU)

import numpy as np
import matplotlib.pyplot as plt
import random

# deteministic random numbers
np.random.seed(0)
random.seed(0)
```

## Cell 2 — Markdown
We have given you the outlines of eggs we want in a numpy file called `target_curves.npy` which you can load using numpy and plot using matplotlib:

## Cell 3 — Code (exec_count=2)
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

## Cell 4 — Markdown
<font color="red" size="8"><b>WARNING: </b></font> If you plan to work on this notebook in colab we would like to warn you that if you do not download any files you save or save the changes separately, upon the runtime being killed either by you closing the tab/browser or by timing out due to inactivity you WILL LOSE YOUR FILES AND WORK. It is highly recommeneded that you make a local python environment to work.

## Cell 5 — Markdown
## Overview
The first portion of this notebook will help you understand the linkage design problem and the way we parameterize each linkage design. It will also discuss deliverables and submission instructions. In the second half of this notebook we provide you with a working implementation that you can use as a baseline. 

**IMPORTANT:** We also provide numerous helper functions in the linkage_utils. We touch on most of these functions in this notebook but some *very useful* functions will be covered in the advanced notebook!

### Project Files and Functions Provided:

All files as well as helper functions and further information can be found in the Github repository of this challenge problem:
[Github Repo](https://github.com/ahnobari/2155-Optimization-Challenge-Problem)

### The Optimization Problem
<div width="100%"><img src="https://i.imgur.com/69xDuZc.png" alt="Path Synthesis Problem" width="60%" style="margin:auto;display:block"></div>

The problem we try to solve in this challenge problem is similar to the one in the optimization demo. The goal in this problem is to design a linkage mechanism that can trace a target curve. In addition to this we also want to minimize material usage. We will discuss the full problem later in the notebook but before we get to that, let us review how we represented linkage mechanisms computationally.


## Cell 6 — Markdown
### Mechanism Representation
Here we will briefly discuss how we can represent linkage mechanisms. In this demonstration we will represent linkage mechanisms as graphs. A graph is represented as a set of nodes (e.g., $N = \{1,2,3,\ldots,N\}$), and set of edges, that describe how nodes are connected to each other (e.g., $E=\{(1,2), (5,2), \ldots \}$). Graphs can also include features associated with each node (e.g., initial positions of joints in our example of mechanisms).

To see how this works take the mechanism below:

<div width="100%"><img src="https://i.ibb.co/qsPC0gC/2021-09-13-0hl-Kleki.png" alt="Path Synthesis Problem" width="40%" style="margin:auto;display:block"></div>


We can see that we have 5 joints, which we will represent as 5 nodes, $N=\{0,1,2,3,4\}$, and we can represent the linkages as edges in the graph, which give us the edges $E=\{(0,2),(1,3),(2,3),(2,4),(3,4)\}$. Note that edges in this graph do not have directions so $(0,2)$ and $(2,0)$ are identical. To keep things neat so we put the smaller index first. Besides the basic graph describing the mechanisms structure, we also see that all nodes have an initial position in 2D space. Nodes $0$ and $1$ are fixed and will remain in place, while the others will move. We will describe these as node features. This means each node will have a 2D initial position and a binary feature describing whether it is fixed or not. We can also see that one of the linkages is highlighed yellow. This is the linkage that is rotated by the actuator. We will also store this information.

In practice the above description boils down to a few (Design) variables in python which we use to describe the mechanism:

- **Initial Positions** `x0`: A 2D array of size $N \times 2$, holding the initial positions of the $N$ joints in the mechanism.
- **Linkages** `edges`: A 2D array of ***integer*** pairs, with shape $E \times 2$ holding the pairs of node indecies from two nodes that are connected to each other by an edge.
- **Fixed Joints** `fixed_joints`: A list of node indices that are fixed.
- **Motor** `motor`: A tuple of integers, which is equivalent to the pair of node indices associated with the actuated edge.

This representation is fine, however sometimes instead of explicitly specifying edges a different approach is used to describe graph connectivity. The so called <b>Adjacency/Connectivity Matrix</b> is another way to represent the edges of a graph. A linkage mechanisms structure can therefore be represented using a matrix of connectivity ($C$), which is similar to the adjacency matrix in graphs (networks). The connectivity matrix for a mechanisms with $N$ nodes is a symmetric $N\times N$ matrix with each row and column representing each node and its connections to other nodes. Take the example below:

<img src="https://transportgeography.org/wp-content/uploads/simple_connectivity_matrix2.png" width="100%" style="max-width: 700px; display:block; margin:auto;" alt="Connectivity Matrix">

For Node A we use the first row/column of the matrix and we see that A is connected to B (which is represented by the second row/column). To represent this connection in the connectivity matrix we will set the value of the $(1,2)$ and $(2,1)$ elements of the matrix to be 1. Similarly any time any nodes $i$ and $j$ are connected with an edge we set the value of the $(i,j)$ and $(j,i)$ elements to be 1 and 0 if the nodes are not connected. We will use this kind of representation in the advanced starter notebook to show how this can help us define a problem for GA that generates the structure of the mechanism as well.

## Cell 7 — Markdown
### Visualizing Mechanisms
In the code base we provide a `MechanismVisualizer` class which you can use to visualize mechanisms. Let's define a four-bar mechanism below and visualize it using this class.

## Cell 8 — Code (exec_count=3)
```python
from LINKS.Visualization import MechanismVisualizer
```

## Cell 9 — Markdown
To use this, first create an instance of the visualizer, then call this instance with your mechanisms representation. Below is an example:

## Cell 10 — Code (exec_count=4)
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

## Cell 11 — Markdown
The `MechanismVisualizer` class will solve the mechanism everytime. But if we want to access the full simulations of the mechanisms we will need to use another class, the `MechanismSolver` class. Below is an example of this:

## Cell 12 — Code (exec_count=5)
```python
from LINKS.Kinematics import MechanismSolver
```

## Cell 13 — Code (exec_count=6)
```python
solver = MechanismSolver(
    device = 'cpu' # device to run the simulation on (cpu or gpu)
)

solution = solver(x0, edges, fixed_joints, motor)

print("The shape of the solution is:", solution.shape)  # Should be (num_joints, timesteps, 2)
```

### Outputs (1)

#### Output 1
```
The shape of the solution is: (5, 200, 2)

```

## Cell 14 — Markdown
We can grab joint 4's solution and visualize it:

## Cell 15 — Code (exec_count=7)
```python
# the target joints solution
traced_curve = solution[4]  # Joint 4's trajectory over time

plt.figure(figsize=(5,5))
plt.plot(traced_curve[:,0], traced_curve[:,1])
plt.axis('equal')
plt.title("Joint 4's Traced Curve")
plt.axis('off')
plt.show()
```

### Outputs (1)

#### Output 1
```
<Figure size 500x500 with 1 Axes>
```

## Cell 16 — Markdown
Now that we have a traced curve, how do we compare with the target curve? To do this we provide another class called `CurveEngine` which has some useful functions which you can use to do this.

## Cell 17 — Code (exec_count=8)
```python
from LINKS.Geometry import CurveEngine
```

## Cell 18 — Markdown
The functions that we most case about are the `compare_curves` and `visualize_comparison`, and `visualize_alignment`.

## Cell 19 — Code (exec_count=9)
```python
curve_processor = CurveEngine(
    normalize_scale=False, # do not normalize the scale of the curve (This would mean we care about the size of the curve too!)
    device='cpu'
)
```

## Cell 20 — Markdown
Now let's visualize the alignment process:

## Cell 21 — Code (exec_count=10)
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

## Cell 22 — Markdown
Note that the visualizations and calculations can be done for a batch of inputs as well:

## Cell 23 — Code (exec_count=12)
```python
distances = curve_processor.compare_curves(
            traced_curve[None].repeat(target_curves.shape[0], axis=0), # repeat the traced curve for each target curve
            target_curves)

print("Distances to all target curves:", distances)
```

### Outputs (1)

#### Output 1
```
Distances to all target curves: [3.3600094 3.2083194 3.400465  3.4069812 3.3541577 3.3789632]

```

## Cell 24 — Code (exec_count=13)
```python
curve_processor.visualize_comparison(
            traced_curve[None].repeat(target_curves.shape[0], axis=0), # repeat the traced curve for each target curve
            target_curves)
```

### Outputs (1)

#### Output 1
```
<Figure size 400x2400 with 6 Axes>
```

## Cell 25 — Markdown
Details on the curve distance calculation can be found at the end of the advanced starter notebook.

## Cell 26 — Markdown
## Instructions


[IMPORTANT] In this project three constraints are set for you:
<ul>
    <li><b>Distance: </b>Any mechanisms output with a distance to the target curve larger than 0.75 will be considered invalid and will not be used to calculate the score of the submissions.</li>
    <li><b>Material Use: </b>Any mechanism with total linkage lengths more than 10.0 will be deemed invalid and will not be used to calculate the score of the submissions.</li>
    <li><b>Mechanism Complexity: </b>For this project we want you to generate mechanisms with no more than 20 joints </li>
</ul>

Submitted mechanisms that violate these constraints will not be scored.

Your performance on the targets will be measured based on the hypervolume calculated from the population with a reference point of 0.75 for distance and 10.0 for material use. Note that we limit the number of mechanisms you are permitted to submit as potential solutions for any given curve to 1000. The closer you get to ideal the higher the score you will receive. The total score for your submission will be the average hypervolume for all target curves.

## Cell 27 — Markdown
## Starter Code: Setting up Multi-Objective Optimization With Constraints Using Pymoo

Now that we have established the details to the project lets set up the problem in Pymoo and run an example. Here we will also discuss how your submissions are going to be evaluated to get your final score. Before that, we will introduce a possible approaches for solving this problem. We will use a GA optimizer availabe through the pymoo package. This approach will be a simple approach to only optimize the positions of a given mechanism (We provide this) and the more comprehensive aproach will be one of allowing the GA to generate the full structure of mechanisms which is discussed in the advanced notebook and not here.

## Cell 28 — Markdown
For this example we provide you with a mechanism structure to use. A bit more complex than the four-bar above to have a better chance at meeting our constraints. This mechanism is in a file called '`starter_mechanism.npy`'. Let's see what this looks like:

## Cell 29 — Code (exec_count=14)
```python
starter_mech = np.load('starter_mechanism.npy',allow_pickle=True).item() #Load mechanism

x0 = starter_mech['x0']
edges = starter_mech['edges']
fixed_joints = starter_mech['fixed_joints']
motor = starter_mech['motor']

plt.figure(figsize=(8,8))
visualizer(x0, edges, fixed_joints, motor, ax=plt.gca())
```

### Outputs (2)

#### Output 1
```
<Axes: >
```

#### Output 2
```
<Figure size 800x800 with 1 Axes>
```

## Cell 30 — Markdown
Besides this starter mechanism, we provide two additional classes called `DifferentiableTools` and `Tools`, which provide you with helpful optimization for this problem. As the names suggest the `DifferentiableTools` provides gradients, if you wish to use gradient based optimization and `Tools` will simply provide the values for the objective functions. Both methods have a `compile` function which will compile the functions for faster and more efficient runs. if you use compile the first run will be slow but then the subsequent runs will speed up. Let's see how they work:

## Cell 31 — Code (exec_count=15)
```python
from LINKS.Optimization import DifferentiableTools, Tools
```

## Cell 32 — Code (exec_count=16)
```python
optimization_tools = Tools(
    device='cpu' # device to run the optimization on
)
optimization_tools.compile() # compile the functions for faster runs

gradient_tools = DifferentiableTools(
    device='cpu' # device to run the optimization on
)
gradient_tools.compile() # compile the functions for faster runs
```

## Cell 33 — Markdown
Let's now see how this mechanism does with target curve 1:

## Cell 34 — Code (exec_count=17)
```python
distance, material = optimization_tools(x0,
                                        edges,
                                        fixed_joints,
                                        motor,
                                        target_curves[0],
                                        target_idx=None # if you do not specify a target node index, it will automatically use the most complex joint (the last joint in our case)
                                    ) 

print(f"Distance to target curve 1: {distance:.4f}")
print(f"Material used: {material:.4f}")
```

### Outputs (1)

#### Output 1
```
Distance to target curve 1: 2.1256
Material used: 14.2079

```

## Cell 35 — Code (exec_count=18)
```python
distance, material, distance_grad, material_grad = gradient_tools(x0,
                                        edges,
                                        fixed_joints,
                                        motor,
                                        target_curves[0],
                                        target_idx=None
                                    )

print(f"Distance to target curve 1: {distance:.4f}")
print(f"Material used: {material:.4f}")
print(f"Distance Gradient:\n {distance_grad}\n")
print(f"Material Gradient:\n {material_grad}\n")
```

### Outputs (1)

#### Output 1
```
Distance to target curve 1: 2.1256
Material used: 14.2079
Distance Gradient:
 [array([ 6.2681365, -1.921999 ], dtype=float32), array([ 1.0770496 , -0.02614211], dtype=float32), array([-4.8808837, -0.815959 ], dtype=float32), array([0., 0.], dtype=float32), array([0., 0.], dtype=float32), array([-3.017497 ,  1.1877289], dtype=float32), array([ 1.9700432, -4.5516434], dtype=float32), array([0., 0.], dtype=float32), array([0., 0.], dtype=float32), array([0., 0.], dtype=float32), array([0.27832413, 2.1568    ], dtype=float32), array([-1.6951733,  3.9712143], dtype=float32)]

Material Gradient:
 [array([-2.1489484, -1.6446065], dtype=float32), array([-3.890974 ,  1.8862138], dtype=float32), array([-1.1027613, -5.6328993], dtype=float32), array([-1.849906 , -0.7601631], dtype=float32), array([3.1135585, 2.032076 ], dtype=float32), array([-3.8731604,  2.1107216], dtype=float32), array([2.5198495, 2.5694587], dtype=float32), array([-0.09414947, -2.855552  ], dtype=float32), array([ 1.2033318, -3.798664 ], dtype=float32), array([3.682773 , 1.5605371], dtype=float32), array([-1.4612309,  4.0732913], dtype=float32), array([3.9016178, 0.459587 ], dtype=float32)]


```

## Cell 36 — Markdown
**NOTE:** Both of these classes accept batches of mechanisms for a single target curve. If you pass batches, this will be alot faster, especially running on gpu. In this code to keep things simple we do `ElementwiseProblem` in pymoo, but pymoo supports vectorized problems with the `Problem` class.

## Cell 37 — Markdown
Now we will use an approach similar to the demo to define our problem. Note that here we will also add consraints to the problem to remove unacceptable mechanisms (Distance above 0.75 and material more than 10.0):

## Cell 38 — Code (exec_count=19)
```python
# pymoo imports
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
```

## Cell 39 — Code (exec_count=20)
```python
PROBLEM_TOOLS = Tools( # we have to define this outside the class due to pymoo deepcopy limitations
            device='cpu' # device to run the optimization on
        )  
PROBLEM_TOOLS.compile() # compile the functions for faster runs

class mechanism_Synthesis_optimization(ElementwiseProblem):

    # When intializing get the mechanism structure which is being optimized
    #(not the initial positions as we are trying to find those)
    def __init__(self, edges, fixed_joints, motor, target_curve):

        # set the problem paramters(n_var = number of parameters(number of nodes * 2),
        # n_obj=number of objectives, xl=lower bound for the parameters, xu=upper bound for the paramters)
        super().__init__(n_var = x0.shape[0] * 2, n_obj=2, n_constr=2, xl=0.0, xu=5.0, elementwise_evaluation=True)

        # Store mechanism information for later
        self.edges = edges
        self.fixed_joints = fixed_joints
        self.motor = motor
        self.N = x0.shape[0]

        # Store the target
        self.target_curve = target_curve
        
    def _evaluate(self, x, out, *args, **kwargs):
        x0 = np.array(x.reshape([self.N,2]))  #reshape the flattened vector that comes in into 2D array
        
        # Simulate
        distance, material = PROBLEM_TOOLS(x0,
                                self.edges,
                                self.fixed_joints,
                                self.motor,
                                self.target_curve,
                                target_idx=None # if you do not specify a target node index, it will automatically use the most complex joint (the last joint in our case)
                            )

        out["F"] = np.array([distance, material])
        out["G"] = out["F"] - np.array([0.75, 10.0])  # Constraints: distance <= 0.75, material <= 10.0
```

## Cell 40 — Markdown
Now we will run NSGA-II on this problem. Note that the code below replaces the sampling of the algorithm with the initial positions of the pre-defined mechanism. This is an example of population initialization. Becuase this mechanism skeleton is bigger the GA usually struggles to find any valid solutions so this will help start with a better population. This will be discussed in more detail in the advanced notebook. For now let's run this on the first curve:

## Cell 41 — Code (exec_count=21)
```python
target_index = 0
target_curve = np.array(target_curves[target_index])
plt.figure(figsize=(5,5))
plt.plot(target_curve[:,0],target_curve[:,1], lw=4)
plt.axis('equal')
plt.axis('off')
plt.title(f'Target Curve {target_index + 1}')
plt.show()
```

### Outputs (1)

#### Output 1
```
<Figure size 500x500 with 1 Axes>
```

## Cell 42 — Code (exec_count=22)
```python
problem = mechanism_Synthesis_optimization(
    edges,
    fixed_joints,
    motor,
    target_curve
)

# Algorith Parameters
population_size = 100
num_generations = 100

# Initialize a population of duplicates
X = x0[None].repeat(100, axis=0).reshape(population_size, -1)

# Setup the algorithm
algorithm = NSGA2(pop_size=population_size,
                    sampling=X,
                    crossover=SBX(prob=1.0, eta=3.0),
                    mutation=PolynomialMutation(eta=3.0, prob=0.05),
                    eliminate_duplicates=True)

results = minimize(problem,
                   algorithm,
                   ('n_gen', num_generations),
                   verbose=True,
                   save_history=True,
                   seed=0,
                  )
```

### Outputs (1)

#### Output 1
```
==========================================================================================
n_gen  |  n_eval  | n_nds  |     cv_min    |     cv_avg    |      eps      |   indicator  
==========================================================================================
     1 |        1 |      1 |  5.5834536552 |  5.5834536552 |             - |             -
WARNING: Mating could not produce the required number of (unique) offsprings!
     2 |       94 |      1 |  4.5967020988 |           INF |             - |             -
     3 |      194 |      1 |  4.5967020988 |  5.9323697245 |             - |             -
     4 |      294 |      1 |  3.8940066099 |  5.3516081309 |             - |             -
     5 |      394 |      1 |  3.8940066099 |  4.9955091929 |             - |             -
     6 |      494 |      1 |  3.1714673042 |  4.5183174920 |             - |             -
     7 |      594 |      1 |  2.8299154043 |  4.0856205249 |             - |             -
     8 |      694 |      1 |  1.6250197887 |  3.5344779754 |             - |             -
     9 |      794 |      1 |  1.4961649179 |  3.0691960835 |             - |             -
    10 |      894 |      1 |  1.3673545122 |  2.6455748940 |             - |             -
    11 |      994 |      1 |  1.1672819853 |  2.2073694372 |             - |             -
    12 |     1094 |      1 |  0.9091387987 |  1.7437829506 |             - |             -
    13 |     1194 |      1 |  0.6577507257 |  1.4114129722 |             - |             -
    14 |     1294 |      1 |  0.5793468952 |  1.0859848213 |             - |             -
    15 |     1394 |      1 |  0.3614096642 |  0.8701426482 |             - |             -
    16 |     1494 |      1 |  0.3614096642 |  0.7067136109 |             - |             -
    17 |     1594 |      1 |  0.3224560022 |  0.5921954751 |             - |             -
    18 |     1694 |      1 |  0.3214524984 |  0.5269742167 |             - |             -
    19 |     1794 |      1 |  0.3069033623 |  0.4456161535 |             - |             -
    20 |     1894 |      1 |  0.3009731770 |  0.3795647669 |             - |             -
    21 |     1994 |      1 |  0.2815200090 |  0.3441515219 |             - |             -
    22 |     2094 |      1 |  0.2595790625 |  0.3195194161 |             - |             -
    23 |     2194 |      1 |  0.2474962473 |  0.3006833541 |             - |             -
    24 |     2294 |      1 |  0.2029805779 |  0.2812562031 |             - |             -
    25 |     2394 |      1 |  0.1434558630 |  0.2551159096 |             - |             -
    26 |     2494 |      1 |  0.1434558630 |  0.2244447392 |             - |             -
    27 |     2594 |      1 |  0.0875844955 |  0.1899457836 |             - |             -
    28 |     2694 |      1 |  0.0390079618 |  0.1598001909 |             - |             -
    29 |     2794 |      1 |  0.0220204592 |  0.1281739330 |             - |             -
    30 |     2894 |      1 |  0.0074658990 |  0.0867534053 |             - |             -
    31 |     2994 |      1 |  0.0000280738 |  0.0476820868 |             - |             -
    32 |     3094 |      1 |  0.000000E+00 |  0.0277040529 |             - |             -
    33 |     3194 |      2 |  0.000000E+00 |  0.0145790458 |  1.3783818802 |         ideal
    34 |     3294 |      2 |  0.000000E+00 |  0.0058610868 |  1.6394087670 |         ideal
    35 |     3394 |      3 |  0.000000E+00 |  0.0008273977 |  0.3163119511 |         ideal
    36 |     3494 |      2 |  0.000000E+00 |  0.000000E+00 |  0.2466602215 |         ideal
    37 |     3594 |      5 |  0.000000E+00 |  0.000000E+00 |  0.5711710187 |         ideal
    38 |     3694 |      9 |  0.000000E+00 |  0.000000E+00 |  0.0180133997 |         ideal
    39 |     3794 |     11 |  0.000000E+00 |  0.000000E+00 |  0.1663793667 |         ideal
    40 |     3894 |      7 |  0.000000E+00 |  0.000000E+00 |  0.4466756194 |         ideal
    41 |     3994 |      8 |  0.000000E+00 |  0.000000E+00 |  0.0256369955 |         ideal
    42 |     4094 |      6 |  0.000000E+00 |  0.000000E+00 |  0.0888650782 |         ideal
    43 |     4194 |      8 |  0.000000E+00 |  0.000000E+00 |  0.1100675067 |             f
    44 |     4294 |      8 |  0.000000E+00 |  0.000000E+00 |  0.0966109600 |         ideal
    45 |     4394 |      9 |  0.000000E+00 |  0.000000E+00 |  0.4012033036 |         ideal
    46 |     4494 |      4 |  0.000000E+00 |  0.000000E+00 |  0.0168299260 |         ideal
    47 |     4594 |      8 |  0.000000E+00 |  0.000000E+00 |  0.2653379832 |         ideal
    48 |     4694 |      8 |  0.000000E+00 |  0.000000E+00 |  0.0138993110 |         ideal
    49 |     4794 |      5 |  0.000000E+00 |  0.000000E+00 |  0.4022496419 |         ideal
    50 |     4894 |      7 |  0.000000E+00 |  0.000000E+00 |  0.0177548217 |         ideal
    51 |     4994 |     10 |  0.000000E+00 |  0.000000E+00 |  0.0733388660 |         ideal
    52 |     5094 |     15 |  0.000000E+00 |  0.000000E+00 |  0.0099030587 |         ideal
    53 |     5194 |     21 |  0.000000E+00 |  0.000000E+00 |  0.0721213410 |         ideal
    54 |     5294 |     17 |  0.000000E+00 |  0.000000E+00 |  0.0519210166 |         ideal
    55 |     5394 |     20 |  0.000000E+00 |  0.000000E+00 |  0.0178164850 |             f
    56 |     5494 |     20 |  0.000000E+00 |  0.000000E+00 |  0.0155216605 |         ideal
    57 |     5594 |     16 |  0.000000E+00 |  0.000000E+00 |  0.0027469350 |         ideal
    58 |     5694 |     16 |  0.000000E+00 |  0.000000E+00 |  0.1306461703 |         ideal
    59 |     5794 |     16 |  0.000000E+00 |  0.000000E+00 |  0.0671228926 |         ideal
    60 |     5894 |     20 |  0.000000E+00 |  0.000000E+00 |  0.0262858202 |         nadir
    61 |     5994 |     13 |  0.000000E+00 |  0.000000E+00 |  0.0887752547 |         ideal
    62 |     6094 |     17 |  0.000000E+00 |  0.000000E+00 |  0.0196910709 |             f
    63 |     6194 |     23 |  0.000000E+00 |  0.000000E+00 |  0.1156939899 |         ideal
    64 |     6294 |     24 |  0.000000E+00 |  0.000000E+00 |  0.0161304407 |             f
    65 |     6394 |     26 |  0.000000E+00 |  0.000000E+00 |  0.1923263500 |         nadir
    66 |     6494 |     33 |  0.000000E+00 |  0.000000E+00 |  0.0051348529 |             f
    67 |     6594 |     28 |  0.000000E+00 |  0.000000E+00 |  0.0076077588 |             f
    68 |     6694 |     34 |  0.000000E+00 |  0.000000E+00 |  0.0064452502 |             f
    69 |     6794 |     31 |  0.000000E+00 |  0.000000E+00 |  0.0408551436 |         ideal
    70 |     6894 |     25 |  0.000000E+00 |  0.000000E+00 |  0.0129357028 |             f
    71 |     6994 |     28 |  0.000000E+00 |  0.000000E+00 |  0.0454419402 |         ideal
    72 |     7094 |     29 |  0.000000E+00 |  0.000000E+00 |  0.0150679676 |             f
    73 |     7194 |     22 |  0.000000E+00 |  0.000000E+00 |  0.0079351969 |         ideal
    74 |     7294 |     26 |  0.000000E+00 |  0.000000E+00 |  0.0349323708 |         ideal
    75 |     7394 |     30 |  0.000000E+00 |  0.000000E+00 |  0.0364853820 |         ideal
    76 |     7494 |     30 |  0.000000E+00 |  0.000000E+00 |  0.0268614449 |         ideal
    77 |     7594 |     29 |  0.000000E+00 |  0.000000E+00 |  0.0252231518 |         ideal
    78 |     7694 |     27 |  0.000000E+00 |  0.000000E+00 |  0.0139754254 |             f
    79 |     7794 |     30 |  0.000000E+00 |  0.000000E+00 |  0.0652165065 |         ideal
    80 |     7894 |     36 |  0.000000E+00 |  0.000000E+00 |  0.0073819693 |         ideal
    81 |     7994 |     32 |  0.000000E+00 |  0.000000E+00 |  0.0145663878 |         ideal
    82 |     8094 |     37 |  0.000000E+00 |  0.000000E+00 |  0.0066705929 |             f
    83 |     8194 |     41 |  0.000000E+00 |  0.000000E+00 |  0.0506256047 |         ideal
    84 |     8294 |     45 |  0.000000E+00 |  0.000000E+00 |  0.1724545146 |         ideal
    85 |     8394 |     44 |  0.000000E+00 |  0.000000E+00 |  0.0045216239 |         ideal
    86 |     8494 |     42 |  0.000000E+00 |  0.000000E+00 |  0.0345288273 |         ideal
    87 |     8594 |     43 |  0.000000E+00 |  0.000000E+00 |  0.0029203118 |             f
    88 |     8694 |      6 |  0.000000E+00 |  0.000000E+00 |  0.6418686327 |         ideal
    89 |     8794 |      1 |  0.000000E+00 |  0.000000E+00 |  0.0399034023 |         ideal
    90 |     8894 |      3 |  0.000000E+00 |  0.000000E+00 |  0.9726731367 |         ideal
    91 |     8994 |      5 |  0.000000E+00 |  0.000000E+00 |  0.5714029326 |         ideal
    92 |     9094 |      1 |  0.000000E+00 |  0.000000E+00 |  0.2504692078 |         ideal
    93 |     9194 |      2 |  0.000000E+00 |  0.000000E+00 |  1.0000000000 |         ideal
    94 |     9294 |      6 |  0.000000E+00 |  0.000000E+00 |  0.6905966382 |         ideal
    95 |     9394 |     11 |  0.000000E+00 |  0.000000E+00 |  0.1332097798 |         ideal
    96 |     9494 |     10 |  0.000000E+00 |  0.000000E+00 |  0.0449620211 |         ideal
    97 |     9594 |     12 |  0.000000E+00 |  0.000000E+00 |  0.0123390829 |         ideal
    98 |     9694 |      5 |  0.000000E+00 |  0.000000E+00 |  0.1213399936 |         ideal
    99 |     9794 |     11 |  0.000000E+00 |  0.000000E+00 |  0.0205946530 |         ideal
   100 |     9894 |     11 |  0.000000E+00 |  0.000000E+00 |  0.0254584556 |         ideal

```

## Cell 43 — Markdown
Let's visualize the non-dominated solution found by the GA:

## Cell 44 — Code (exec_count=23)
```python
from LINKS.Visualization import GAVisualizer
```

## Cell 45 — Code (exec_count=24)
```python
from pymoo.indicators.hv import HV
ga_visualizer = GAVisualizer()

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
Hyper Volume ~ 1.204261

```

#### Output 2
```
<Figure size 600x600 with 1 Axes>
```

## Cell 46 — Markdown
Now Let's visualize the best solution for distance

## Cell 47 — Code (exec_count=25)
```python
if not results.X is None:
    if type(results.X)!=dict:
        best_idx = np.argmin(results.F[:,0])
        best_sol = results.X[best_idx].reshape(-1,2) # get the solution with the lowest distance to target curve (and reshape it)
    else:
        best_sol = results.X.reshape(-1,2)
else:
    print('Did Not Find Solutions!!')
```

## Cell 48 — Code (exec_count=26)
```python
plt.figure(figsize=(8,8))
visualizer(best_sol, edges, fixed_joints, motor, ax=plt.gca())

traced_curve = solver(best_sol, edges, fixed_joints, motor)[-1]
curve_processor.visualize_comparison(
    traced_curve,
    target_curves[target_index]
)

print(f'Distance: {results.F[best_idx,0]:.4f}, Material: {results.F[best_idx,1]:.4f}')
```

### Outputs (3)

#### Output 1
```
Distance: 0.3809, Material: 7.6436

```

#### Output 2
```
<Figure size 800x800 with 1 Axes>
```

#### Output 3
```
<Figure size 400x400 with 1 Axes>
```

## Cell 49 — Markdown
Now let's visualize the best solution for material

## Cell 50 — Code (exec_count=27)
```python
if not results.X is None:
    if type(results.X)!=dict:
        best_idx = np.argmin(results.F[:,1])
        best_sol = results.X[best_idx].reshape(-1,2) # get the solution with the lowest distance to target curve (and reshape it)
    else:
        best_sol = results.X.reshape(-1,2)
else:
    print('Did Not Find Solutions!!')
```

## Cell 51 — Code (exec_count=28)
```python
plt.figure(figsize=(8,8))
visualizer(best_sol, edges, fixed_joints, motor, ax=plt.gca())

traced_curve = solver(best_sol, edges, fixed_joints, motor)[-1]
curve_processor.visualize_comparison(
    traced_curve,
    target_curves[target_index]
)

print(f'Distance: {results.F[best_idx,0]:.4f}, Material: {results.F[best_idx,1]:.4f}')
```

### Outputs (3)

#### Output 1
```
Distance: 0.4824, Material: 6.6503

```

#### Output 2
```
<Figure size 800x800 with 1 Axes>
```

#### Output 3
```
<Figure size 400x400 with 1 Axes>
```

## Cell 52 — Markdown
### Submission Format
We want you to submit a single numpy file which includes your solutions to each of the 6 problems. This numpy file will be a standard python dictionary with following format:

```python
submission = {
    'Problem 1': [
        {
            'x0': np.ndarray of shape (N, 2),
            'edges': np.ndarray of shape (E, 2),
            'fixed_joints': np.ndarray of shape (F,),
            'motor': np.ndarray of shape (2,),
            'target_joint': int index of the target node

        },
        ...
    ]
    'Problem 2': [...],
    ...
    'Problem 6':[...],
}
```

To make things a bit easier for you, we have provided a few helper functions which will create this dictionary for you. Also note that if you do not provide `target_joint`, during evaluation we will automatically pick the most complex joint (highest index) for evaluation, so be careful to include that if your optimizer picked joints. Finally, if any of the problems are not in your submission dictionary or if the list in the dictionary is empty, a hypervolume of 0 will be assigned to that problem. Your final score is the average hypervolume across all 6 problems.

Let's use the solution of problem 1 that we just made to make a submission:

## Cell 53 — Code (exec_count=30)
```python
from LINKS.CP import make_empty_submission, evaluate_submission

submission = make_empty_submission()

print("Empty Submission:", submission)
```

### Outputs (1)

#### Output 1
```
Empty Submission: {'Problem 1': [], 'Problem 2': [], 'Problem 3': [], 'Problem 4': [], 'Problem 5': [], 'Problem 6': []}

```

## Cell 54 — Code (exec_count=31)
```python
# add out population to submission
if not results.X is None:
    for i in range(results.X.shape[0]):
        x0_member = results.X[i].reshape(-1,2)
        
        mech = {
            'x0': x0_member,
            'edges': edges,
            'fixed_joints': fixed_joints,
            'motor': motor,
            'target_joint': None # None means it will automatically use the most complex joint (the last joint in our case)
        }
        
        submission['Problem 1'].append(mech)
```

## Cell 55 — Markdown
Now let's get our score:

## Cell 56 — Code (exec_count=32)
```python
evaluate_submission(submission)
```

### Outputs (1)

#### Output 1
```
{'Overall Score': 0.20071005683320445,
 'Score Breakdown': {'Problem 1': 1.2042603409992267,
  'Problem 2': 0.0,
  'Problem 3': 0.0,
  'Problem 4': 0.0,
  'Problem 5': 0.0,
  'Problem 6': 0.0}}
```

## Cell 57 — Markdown
We can also visualize the non-dominated samples in the submission population (it will take a while since we have alot of samples to visualize):

## Cell 58 — Code (exec_count=33)
```python
ga_visualizer.plot_pareto_efficient(
    F=results.F,
    population=submission['Problem 1'],
    target_curve=target_curve,
    objective_labels=['Distance', 'Material']
)
```

### Outputs (1)

#### Output 1
```
<Figure size 1500x5500 with 33 Axes>
```

## Cell 59 — Markdown
In your submission we expect a single numpy file. Below is some code to save this submission as a numpy file:

## Cell 60 — Code (exec_count=34)
```python
np.save('my_submission.npy', submission) # This will save your submission in the current directory
```

## Cell 61 — Markdown
You can call the `evaluate_submission` functions with file paths too:

## Cell 62 — Code (exec_count=35)
```python
evaluate_submission(
    submission='my_submission.npy',
    target_curves='target_curves.npy'
)
```

### Outputs (1)

#### Output 1
```
{'Overall Score': 0.20071005683320445,
 'Score Breakdown': {'Problem 1': 1.2042603409992267,
  'Problem 2': 0.0,
  'Problem 3': 0.0,
  'Problem 4': 0.0,
  'Problem 5': 0.0,
  'Problem 6': 0.0}}
```

## Cell 63 — Markdown
When you are submitting to the scoreboard you will need to upload the submission numpy file.

## Cell 64 — Markdown
### Let's get a full submission going
Now we will run the GA for each target curve seperately and get a full submission going:

## Cell 65 — Code (exec_count=36)
```python
submission = make_empty_submission()
full_results = submission.copy()

population_size = 200
num_generations = 50

for i in range(len(target_curves)):
    
    print(f"Optimizing for Problem {i+1}, Please Wait...")
    
    problem = mechanism_Synthesis_optimization(
        edges,
        fixed_joints,
        motor,
        target_curves[i]
    )
    
    X = x0[None].repeat(population_size, axis=0).reshape(population_size, -1)
    # Setup the algorithm
    algorithm = NSGA2(pop_size=population_size,
                        sampling=X,
                        crossover=SBX(prob=1.0, eta=3.0),
                        mutation=PolynomialMutation(eta=3.0, prob=0.9),
                        eliminate_duplicates=True)

    results = minimize(problem,
                    algorithm,
                    ('n_gen', num_generations),
                    verbose=False,
                    save_history=True,
                    seed=123,
                    )
    
    if results.X is None:
        print('Did Not Find Solutions!!')
    else:
        for j in range(results.X.shape[0]):
            x0_member = results.X[j].reshape(-1,2)
            
            mech = {
                'x0': x0_member,
                'edges': edges,
                'fixed_joints': fixed_joints,
                'motor': motor,
                'target_joint': None # None means it will automatically use the most complex joint (the last joint in our case)
            }

            submission[f'Problem {i+1}'].append(mech)
            full_results[f'Problem {i+1}'] = results


# now let's evaluate the pareto efficient solutions for all problems
plt.subplots(2, 3, figsize=(15, 10))
for i in range(len(target_curves)):
    if isinstance(full_results[f'Problem {i+1}'], list) or full_results[f'Problem {i+1}'].X is None:
        print(f'Did Not Find Solutions for Problem {i+1}!!')
        plt.subplot(2, 3, i+1)
        plt.text(0.5, 0.5, 'No Solutions Found', horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        continue
    ga_visualizer.plot_HV(
        F=full_results[f'Problem {i+1}'].F,
        ref=np.array([0.75, 10.0]),
        ax=plt.subplot(2, 3, i+1),
        objective_labels=['Distance', 'Material']
    )
    plt.title(f'Problem {i+1} Hypervolume')
```

### Outputs (2)

#### Output 1
```
Optimizing for Problem 1, Please Wait...
Optimizing for Problem 2, Please Wait...
Optimizing for Problem 3, Please Wait...
Optimizing for Problem 4, Please Wait...
Optimizing for Problem 5, Please Wait...
Optimizing for Problem 6, Please Wait...

```

#### Output 2
```
<Figure size 1500x1000 with 6 Axes>
```

## Cell 66 — Code (exec_count=37)
```python
evaluate_submission(submission)
```

### Outputs (1)

#### Output 1
```
{'Overall Score': 1.3392431216636094,
 'Score Breakdown': {'Problem 1': 2.0762848605094746,
  'Problem 2': 1.6275143085831445,
  'Problem 3': 1.2924261160987953,
  'Problem 4': 1.3881146582029658,
  'Problem 5': 0.8565706125076957,
  'Problem 6': 0.7945481740795799}}
```

## Cell 67 — Markdown
Now we can save this submission and submit it to the leader board.

## Cell 68 — Code (exec_count=38)
```python
np.save('my_full_submission.npy', submission) # This will save your submission in the current directory
```

## Cell 69 — Markdown
<H1>Leader Board: </H1>
You can submit your numpy files to the leaderboard and see how you are doing compared to others. 
The leaderboard can be found at:

[https://design-ai.mit.edu/CP1/](https://design-ai.mit.edu/CP1/)

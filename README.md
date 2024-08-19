# Graph-based-exploration
This repo contains some experiments of Q function approximation using graph Laplacian-based features on Four room grid environment.
## The environment
The environment is made of 53 navigable states and 47 wall states. The possible actions are 4 (up, down, right and left), the transition from one state to another is deterministic when choosing an action. If the action leads to a wall state the agent stays still in the same state. The tile corrsponding to the goal (represented with a star in the grid) is an absorbing state.
The reward is -1 until the goal tile is reached, the transition towards the goal tile has reward 0.
The grid can be plotted as follows:
```python
from four_room_grid import Four_room_grid
grid = Four_room_grid()
grid.plot_grid()
```
![A visual representation of the Four Room Grid environment](images/four_room_grid.png)

## State-graph
To state-graph is the graph where each node is a state. The edge from one state to another exists if there exists an action that, if performed from the first state, has probability > 0 of reaching the second state. The graph can be created and plotted as follows:
```python
from four_room_grid import Four_room_grid
grid = Four_room_grid()
g,A = grid.create_state_graph() # g is a networkx.Graph(), A is the adjacency matrix
grid.plot_state_graph()
```
![A visual representation of the state-graph](images/state_graph.png)

The laplacian can be calculated as follows:

```python
from four_room_grid import Four_room_grid, Policy
grid = Four_room_grid()
g,A = grid.create_state_graph()
L = grid.laplacian_state() # combinatorial laplacian
norm_L = grid.normalized_laplacian_state() # normalized laplacian
```

## State-action graph
The state action graph has nodes corresponding to each state-action pair.
Two experiments in particular were performed:
  - Unweighted and undirected graph, similar to the state action graph, in this way the laplacian is calculated as usual as D-A.
  - Weighted and directed graph as in 2005, Chung, F. Laplacians and the Cheeger Inequality for Directed Graphs. and the precautions as in Osentoski, Mahadevan, 2007, Learning State-Action    Basis functions for Hierarchical MDPs (for more details check the implementation in ```four_room_grid.py```)

To plot the unweighted and undirected graph:

```python
from four_room_grid import Four_room_grid
grid = Four_room_grid()
g,A = grid.create_state_action_graph()
grid.plot_state_action_graph()
```
![Visualization of the unweighted and undirected state-action graph](images/unweighted_undirected_state_action_graph.png)

To plot the weighted and directed graph:
```python
from four_room_grid import Four_room_grid, Policy
grid = Four_room_grid()
uniform_policy = Policy()
grid.set_policy(uniform_policy)
G,W = grid.create_state_action_graph(weighted = True)
grid.plot_state_action_graph()
```
![Visualization of the weighted and directed state-action graph](images/weighted_directed_state_action_graph.png)

To approximate the Q function using linear function approximation with the eigenvectors of the state-action graph:
```python
import numpy as np
from four_room_grid import Four_room_grid

def approximate_Q_pi_using_laplacian(k, target_pi, normalized = False, weight_policy = None, return_Q_target = False):
    grid = Four_room_grid()
    Q_target = target_pi.calculate_Q_pi(as_vector= True) # true q function
    if weight_policy is None:
        if normalized:
            L = grid.normalized_laplacian_state_action() 
        else:
            L = grid.laplacian_state_action()
    else:
        grid.set_policy(weight_policy)
        if normalized:
            L = grid.normalized_laplacian_state_action(weighted = True)
        else:
            L = grid.laplacian_state_action(weighted= True)        

    U, _, _ = np.linalg.svd(L)
    phi = U[:,-k:] # using the eigenvectors associated to the lowest eigenvalues
    weights = np.linalg.solve(phi.T@phi, phi.T@Q_target) # approximation using OLS
    Q_predicted = phi @ weights
    
    if return_Q_target:
        return Q_predicted, Q_target
    else:
        return Q_predicted
```
In ```example.ipynb``` there are some examples on the approximation of the Q function using the two different graph representations for different policies.





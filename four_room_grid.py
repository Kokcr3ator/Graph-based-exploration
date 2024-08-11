from state import State
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
from itertools import product         
    
        
class Four_room_grid:

    def __init__(self, goal_reward = 0):
        self.initialize_states()
        self.initialize_actions()
        self.initialize_transition_matrix()
        self.initialize_reward(goal_reward)
        self.state_graph = None
        self.state_action_graph = None
        self.policy = None
        
    def initialize_states(self):
        """
        Initilizes the states as walls or rewards (normal is default). Navigable states are the tiles in which the agent can be in.
        idx_to_state is a map between a state (i,j) to an index. state_to_idx is a map between an index and a state (i,j).
        Those are used to define the transition probability matrix.
        """
        self.n_rows = int(np.sqrt(100))
        self.n_cols = int(np.sqrt(100))
        self.states = np.array([np.array([State(i,j) for j in range(self.n_cols)]) for i in range(self.n_rows)])
        
        for j in range(self.n_cols):
            self.states[0,j].state_type = 'wall'
            self.states[-1,j].state_type = 'wall'
            self.states[j,0].state_type = 'wall'
            self.states[j,-1].state_type = 'wall'
        
        self.states[1,5].state_type = 'wall'
        self.states[2,5].state_type = 'wall'
        self.states[4,5].state_type = 'wall'
        self.states[5,5].state_type = 'wall'
        self.states[6,5].state_type = 'wall'
        self.states[7,5].state_type = 'wall'

        self.states[4,1].state_type = 'wall'
        self.states[4,3].state_type = 'wall'
        self.states[4,4].state_type = 'wall'
        self.states[4,5].state_type = 'wall'
        self.states[4,6].state_type = 'wall'
        self.states[4,8].state_type = 'wall'
        self.states[4,9].state_type = 'wall'

        self.states[8,1].state_type = 'goal'
        self.goal_state = (8,1)

        self.navigable_states = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) if self.states[i, j].state_type != 'wall']

        self.n_navigable_states = len(self.navigable_states)

        self.state_to_index = {state: idx for idx, state in enumerate(self.navigable_states)}
        self.index_to_state = {idx: state for idx, state in enumerate(self.navigable_states)}

        

    def initialize_actions(self):
        """
        Initilizes the possible actions 'up', 'down', 'left' and 'right', the keys are the resulting shift vector
        """
        self.actions = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        self.n_actions = len(self.actions)

    def initialize_transition_matrix(self):
        """
        Initilizes the transition matrix, which is a 3 dimensional array of shape |S|x|A|x|S|, 
        where entry (i,j,k) is the probability of transition from i to k performing action j.
        Only navigable states are taking into consideration. The goal state is modeled as an absorbing state.
        """
        
        T = np.zeros((self.n_navigable_states, self.n_actions, self.n_navigable_states))
        
        for (i,j) in self.navigable_states:
            current_state = self.state_to_index[(i, j)]
            
            for action_index, (action, (di, dj)) in enumerate(self.actions.items()):
                new_i, new_j = i + di, j + dj
                
                if (new_i, new_j) in self.state_to_index:
                    new_state = self.state_to_index[(new_i, new_j)]
                    T[current_state, action_index, new_state] = 1
                else:
                    # stay there
                    T[current_state, action_index, current_state] = 1

        for action in range(self.n_actions):
            T[self.state_to_index[(8,1)],action,:] = np.zeros(self.n_navigable_states)
            T[self.state_to_index[(8,1)],action,self.state_to_index[(8,1)]] = 1

        self.transition_matrix = T

    def initialize_reward(self, goal_reward):
        r = np.zeros((self.n_navigable_states, self.n_actions)) - 1

        r[self.state_to_index[(7,1)],3] = goal_reward # go right
        r[self.state_to_index[(8,2)],1] = goal_reward # go down
        self.reward_matrix = r


    def plot_grid(self, policy = None):
        fig, ax = plt.subplots()

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                state = self.states[i, j]
                if state.state_type == 'wall':
                    color = 'blue'
                elif state.state_type == 'goal':
                    color = 'white'
                else:
                    color = 'white'

                rect = Rectangle((i, j), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                
                if state.state_type == 'goal':
                    ax.plot(i + 0.5, j + 0.5, 'r*', markersize=15)

        if policy is not None:
            for (idx,state) in enumerate(policy):
                i,j = self.index_to_state[idx]
                for (idx_action, prob_action) in enumerate(state):
                    if prob_action > 0:
                        if idx_action == 0:  # up
                            dx, dy = 0, 0.3
                        elif idx_action == 1:  # down
                            dx, dy = 0, -0.3
                        elif idx_action == 2:  # left
                            dx, dy = -0.3, 0
                        elif idx_action == 3:  # right
                            dx, dy = 0.3, 0

                        ax.arrow(i + 0.5, j + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.show()

    def create_state_graph(self):
        nodes = list(range(self.n_navigable_states))
        edges =[]
        for source_node in nodes:
            for dest_node in nodes:
                if source_node == dest_node: # no self-loops
                    continue
                else:
                    if sum(self.transition_matrix[source_node,:,dest_node]) > 0: # there exists an action which takes me from source state to dest state
                        edges.append((source_node,dest_node))
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        A = nx.adjacency_matrix(G, nodelist=nodes).toarray()
    
        self.state_graph = G
        self.state_adjacency_matrix = A
        return (self.state_graph,self.state_adjacency_matrix)
    
    def plot_state_graph(self):
        if self.state_graph is None:
            _,_ = self.create_state_graph()
        
        plt.figure(figsize=(4, 3)) 
        nx.draw(self.state_graph, self.index_to_state, with_labels=True, node_size=350, node_color="skyblue", font_size=8, font_weight="bold", edge_color="gray")
        plt.show()

    def laplacian_state(self):
        """
        Given the graph with states as nodes, calculates the combinatorial Laplacian as L = D-A
        """
        if self.state_graph is None:
            _,_ = self.create_state_graph()
        
        D = np.diag([sum(row) for row in self.state_adjacency_matrix])
        L = D-self.state_adjacency_matrix

        return L
    
    def normalized_laplacian_state(self):
        """
        Given the graph with states as nodes, calculates the normalized Laplacian as L = D^(-1/2)(D-A)D^(-1/2)
        """        
        if self.state_graph is None:
            _,_ = self.create_state_graph()       

        D = np.diag([sum(row) for row in self.state_adjacency_matrix]) 
        invsqrtD = np.diag([1/np.sqrt(sum(row)) for row in self.state_adjacency_matrix])
        
        norm_L = invsqrtD @ (D - self.state_adjacency_matrix) @ invsqrtD

        return norm_L

    def create_state_action_graph(self, weighted = False):
        """
        INPUT: weighted (bool)
            - if True: the resulting graph will be directed and weighted given a policy (which has to be set using set_policy())

            - if False: the resulting graph will be undirected and unweighted

        Creates the graph with (state, action) tuples as nodes. A node (s,a) will have an edge to (s',a') if from s, by performing action a
        s' can be reached and a' can be performed from s'.

        If weighted is True the weight matrix will be created by taking the inverse of the probability of transitioning given policy pi as in
        Osentoski, Mahadevan, 2007, Learning State-Action Basis functions for Hierarchical MDPs.
        "We use the inverse because the Laplacian treats W as a similarity matrix rather than a distance matrix"
        
        """
        states = list(range(self.n_navigable_states))
        actions = list(range(self.n_actions))
        nodes = list(product(states, actions)) # Cartesian product between states and actions
        self.state_action_to_idx = {node:idx for idx,node in enumerate(nodes)}
        edges =[]
        for source_node in nodes:
            for dest_node in nodes:
                source_state, source_action = source_node
                dest_state, dest_action = dest_node
                if source_state == dest_state:
                    continue
                else:
                    if self.transition_matrix[source_state, source_action, dest_state] > 0 and sum(self.transition_matrix[dest_state, dest_action, :]) > 0:
                        edges.append((source_node,dest_node))

        if not weighted:
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for edge in edges:
                G.add_edge(edge[0], edge[1])
            A = nx.adjacency_matrix(G, nodelist=nodes).toarray()
        
            self.state_action_graph = G
            self.state_action_adjacency_matrix = A
            return (self.state_action_graph,self.state_action_adjacency_matrix)
        else:
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            if self.policy is None:
                raise ValueError("Set policy first")
            
            for edge in edges:
                source_state, source_action = edge[0]
                dest_state, dest_action = edge[1]

                if self.policy.matrix[source_state,source_action] and self.policy.matrix[dest_state,dest_action] > 0:
                    weight = 1/self.policy.matrix[source_state,source_action] * 1/self.policy.matrix[dest_state,dest_action]
                else:
                    weight = 0

                G.add_edge(edge[0], edge[1], weight = weight)

            self.state_action_graph = G
            A = nx.adjacency_matrix(G, nodelist=nodes).toarray()
            self.state_action_directed_adjacency_matrix = A
            self.weight_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray()

            return (self.state_action_graph,self.weight_matrix)
    
    def plot_state_action_graph(self):

        if self.state_action_graph is None:
            _,_ = self.create_state_action_graph()
        
        plt.figure(figsize=(8, 6)) 
        nx.draw(self.state_action_graph, node_size=20, node_color="blue", edge_color="gray")
        plt.show()

    def laplacian_state_action(self, weighted = False):
        """
        INPUT: weighted (bool)
            - if True: The combinatorial Laplacian is calculated for a directed and weighted graph using the formula in 
                        2005, Chung, F. Laplacians and the Cheeger Inequality for Directed Graphs and the precautions as in
                        Osentoski, Mahadevan, 2007, Learning State-Action Basis functions for Hierarchical MDPs.
                        "With probability η the agent acts according to the transition matrix P and with probability 1 - nu teleports
                        to any other vertex in the graph uniformly at random." (similar to page-rank algorithm)

            - if False: The combinatorial Laplacian is calculated for a unweighted, undirected graph as L = D-A

        Computes the combinatorial Laplacian for the graph with state action pairs as nodes. 
        """
    
        _,_ = self.create_state_action_graph(weighted)

        if not weighted:
            D = np.diag([sum(row) for row in self.state_action_adjacency_matrix])
            L = D-self.state_action_adjacency_matrix

        else:

            if self.weight_matrix is None:
                raise ValueError("Set policy first")
            
            else:

                # Compute P as D^(-1)W
                W = np.array([row if sum(row) > 0 else np.ones(len(self.weight_matrix)) for row in self.weight_matrix])
                invD = np.diag([1/sum(row) for row in W])
                P = invD@W

                # Teleport to any node at random with probability nu -> the resulting graph is strongly connected (in this case a clique)
                nu = 0.01
                teleport_P = np.ones(P.shape)
                teleport_P = np.array([np.array(row)/sum(row) for row in teleport_P])
                P = (1-nu)*P + nu*teleport_P

                # Compute the Perron eigenvector (left eigenvector associated to the eigenvalue = 1) that will be the stationary distribution over nodes
                eigenvals, eigenvects = np.linalg.eig(P.T)
                close_to_1_idx = np.isclose(eigenvals,1)
                target_eigenvect = eigenvects[:,close_to_1_idx]
                target_eigenvect = target_eigenvect[:,0]

                # The sum of the components should be 1
                stationary_distrib = target_eigenvect / sum(target_eigenvect)

                # The eigenvector has real components but for numerical reasons take the real part and let the smallest elements be zero
                stationary_distrib = stationary_distrib.real 
                stationary_distrib = stationary_distrib/sum(stationary_distrib)
                psi = np.diag(stationary_distrib)

                L = psi - (psi@P + P.T@psi)/2

        return L
    
    def set_policy(self, policy):
        self.policy = policy


    
    def normalized_laplacian_state_action(self, weighted = False):
        """
        INPUT: weighted (bool)
            - if True: The normalized Laplacian is calculated for a directed and weighted graph using the formula in 
                        2005, Chung, F. Laplacians and the Cheeger Inequality for Directed Graphs and the precautions as in
                        Osentoski, Mahadevan, 2007, Learning State-Action Basis functions for Hierarchical MDPs.
                        "With probability η the agent acts according to the transition matrix P and with probability 1 - nu teleports
                        to any other vertex in the graph uniformly at random." (similar to page-rank algorithm)


            - if False: The normalized Laplacian is calculated for a unweighted, undirected graph as L = D^(-1/2)(D-A)D^(-1/2)

        Computes the normalized Laplacian for the graph with state action pairs as nodes. 
        """

        _,_ = self.create_state_action_graph(weighted)       
        if not weighted:
            D = np.diag([sum(row) for row in self.state_action_adjacency_matrix]) 
            invsqrtD = np.diag([1/np.sqrt(sum(row)) for row in self.state_action_adjacency_matrix])
            
            norm_L = invsqrtD @ (D - self.state_action_adjacency_matrix) @ invsqrtD
        else: 
            if self.weight_matrix is None:
                raise ValueError("Weights not provided")
            else:

                # Compute P as D^(-1)W
                W = np.array([row if sum(row) > 0 else np.ones(len(self.weight_matrix)) for row in self.weight_matrix])
                invD = np.diag([1/sum(row) for row in W])
                P = invD@W

                # Teleport to any node at random with probability nu -> the resulting graph is strongly connected (in this case a clique)
                nu = 0.01
                teleport_P = np.ones(P.shape)
                teleport_P = np.array([np.array(row)/sum(row) for row in teleport_P])
                P = (1-nu)*P + nu*teleport_P

                # Compute the Perron eigenvector (left eigenvector associated to the eigenvalue = 1) that will be the stationary distribution over nodes
                eigenvals, eigenvects = np.linalg.eig(P.T)
                close_to_1_idx = np.isclose(eigenvals,1)
                target_eigenvect = eigenvects[:,close_to_1_idx]
                target_eigenvect = target_eigenvect[:,0]
                
                # The elements of the stationary distribution need to sum to 1
                stationary_distrib = target_eigenvect / sum(target_eigenvect)
                stationary_distrib = stationary_distrib.real
                stationary_distrib = stationary_distrib/sum(stationary_distrib)
                sqrtpsi = np.diag([np.sqrt(psi) for psi in stationary_distrib])
                invsqrtpsi = np.diag([1/np.sqrt(psi) for psi in stationary_distrib])

                I = np.eye(P.shape[0])

                norm_L = I - (sqrtpsi @ P @ invsqrtpsi + invsqrtpsi @ P.T @ sqrtpsi)/2  

        return norm_L
    




class Policy(Four_room_grid):

    def __init__(self, matrix = None, goal_reward = 0):
        super().__init__(goal_reward = goal_reward)
        if matrix is None:
            pi_uniform = np.zeros((self.n_navigable_states,self.n_actions)) + 1/4
            self.matrix = pi_uniform
        else:
            if matrix.shape == (self.n_navigable_states,self.n_actions):
                self.matrix = matrix
            else:
                raise ValueError('Incorrect shape')
            
        self.calculate_r_pi()
        self.calculate_P_pi()

    def calculate_r_pi(self):
        """
        The reward vector associated to policy pi is now a vector of length |S| where r(s) = sum(a in A) r(s,a)* pi(s|a) 
        """
        r_pi = self.matrix * self.reward_matrix # element-wise product
        r_pi = np.sum(r_pi, axis = 1)
        self.r_pi = r_pi
    
    def calculate_P_pi(self):
        """
        From the transition matrix(tensor) |S|x|A|x|S| compute the transition probability matrix between states |S|x|S|
        """
        P_pi = np.zeros((self.n_navigable_states,self.n_navigable_states))
        for i in range(self.n_navigable_states):
            P_pi[i,:] = self.transition_matrix[i,:,:].T @ self.matrix[i,:]
        
        self.P_pi = P_pi
        return self.P_pi

    def calculate_V_pi(self, gamma = 0.99):
        """
        Calculate exactly the value function associated to the policy using the Bellman equation
        """
        I = np.eye(self.n_navigable_states,self.n_navigable_states)
        V_pi = np.linalg.solve(I - gamma*self.P_pi, self.r_pi)
        return V_pi
    
    def calculate_Q_pi(self, gamma = 0.99, as_vector = False):
        """
        INPUT: as_vector (bool)
            -if True: the output will be an |S||A| long vector

            -if False: the output will be an |S|x|A| matrix

        Calculate exactly the action-value function associated to the policy using the Bellman equation
        """
        V_pi = self.calculate_V_pi(gamma)
        Q_pi = self.reward_matrix.copy()
        rows,cols = Q_pi.shape
        for i in range(rows):
            for j in range(cols):
                Q_pi[i][j] += gamma*np.dot(self.transition_matrix[i,j,:],V_pi)

        if as_vector:
            if self.state_action_graph is None:
                _,_ = self.create_state_action_graph()
            
            Q_pi_vector = np.zeros(rows*cols)
            for i in range(rows):
                for j in range(cols):
                    Q_pi_vector[self.state_action_to_idx[(i,j)]] = Q_pi[i,j]
            Q_pi = Q_pi_vector
        return Q_pi 




        

                       

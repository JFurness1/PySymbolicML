import math
import numpy as np
from scipy.optimize import differential_evolution
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class Symbol:
    """
    Main class powering the Symbolic expression.

    Forms a node as part of a tree structure of zero, first, or second order operations
    requiring 0, 1, or 2, arguments respectively. These arguments can be constants, descriptors, 
    or a further Symbol (which may also have further child symbols etc.)

    To add new operations you must add their definition as a method of this class, enter them with a name
    in the opdict that is set during __init__, add it to the appropriate classification list, and add an entry
    for the __str__ representation


    self.cval holds the value for the constant type symbol

    self.descriptor_idx holds the index so that the correct value can be obtained from the list of descriptors. 
    """

    # lists classifying the operations
    zero_order = ['const', 'descriptor']
    first_order = ['sqrt', 'pow2', 'pow3']
    second_order = ['+', '-', '*', '/']

    # How deep the graph can go
    MAX_DEPTH = 100
    # How fast to bias towards zeroth order symbols as we go deeper in the graph
    DECAY = 0.05

    # Chance for a new zero order symbol to be a constant vs. a descriptor
    CONST_CHANCE = 0.3

    def __init__(self, op: str, n_descriptors: int, depth: int, cval=None, descriptor_idx=None):
        self.op = op
        self.n_descriptors = n_descriptors
        self.depth = depth
        self.cval = cval
        self.descriptor_idx = descriptor_idx
        assert not (self.op == 'descriptor' and descriptor_idx is None), "None index for descriptor"
        assert not (self.op == 'const' and cval is None), "None value for constant"

        # List containing references to child symbols. Presently should have max 2 members
        self.symbols = []

        self.opdict = {
            'const': self.const,
            'descriptor': self.descriptor,

            'sqrt': self.sqrt,
            'pow2': self.pow2,
            'pow3': self.pow3,
            'bound': self.bound,

            '+': self.add,
            '-': self.subtract,
            '*': self.multiply,
            '/': self.divide
        }
        
        # Caches counts of how many child symbols this symbol has
        self.dependents = 0

    def set_random_inputs(self):
        """
        Populate the symbol's list of childeren recursively with new random symbols.
        """
        if self.op in self.zero_order:
            return
        elif self.op in self.first_order:
            out = Symbol.new_random_symbol(self.n_descriptors, self.depth + 1)
            out.set_random_inputs()
            self.symbols = [out]
        elif self.op in self.second_order:
            out1 = Symbol.new_random_symbol(self.n_descriptors, self.depth + 1)
            out1.set_random_inputs()
            out2 = Symbol.new_random_symbol(self.n_descriptors, self.depth + 1)
            out2.set_random_inputs()
            self.symbols = [out1, out2]

    @staticmethod
    def new_random_symbol(n_descriptors: int, depth: int):
        """
        Generate a new random symbol with child symbols as appropriate.

        Will bias towards selecting zero order symbols as the depth increases to prevent recursion depth getting out of hand.
        """
        choice = random.random() < math.exp(-Symbol.DECAY*depth)
        if choice and depth + 1 < Symbol.MAX_DEPTH:
            op = random.choice(Symbol.first_order + Symbol.second_order)
            out = Symbol(op, n_descriptors, depth + 1)
        else:
            if random.random() < Symbol.CONST_CHANCE:
                out = Symbol('const', n_descriptors, depth + 1, cval=random.random())
            else:
                out = Symbol('descriptor', n_descriptors, depth + 1, descriptor_idx=random.randint(0, n_descriptors - 1))
            
        out.set_random_inputs()
        return out

    def clone(self):
        """
        Return a new Symbol object that is a direct copy of the present one.
        """
        out = Symbol('const', self.n_descriptors, self.depth, cval=1.0)
        out.copy(self)
        return out

    def copy(self, example):
        """
        Given an example Symbol, set this symbol to be identical.
        """
        self.op = example.op
        self.symbols = []+example.symbols
        self.depth = example.depth
        self.dependents = example.dependents
        self.cval = example.cval
        self.descriptor_idx = example.descriptor_idx
        self.n_descriptors = example.n_descriptors

    def eval(self, descriptors: list):
        """
        Recursively calculate the value of this symbol, given descriptors provided. 
        """
        return self.opdict[self.op](descriptors)

    def set_constants(self, constants, working_idx):
        """
        Given the list of constants, recursively set the self.cval with the appropriate constant.
        Keeps track of position in the constants list by recursively passing and receiving working_idx,
        incrementing working_idx when a constant is assigned.
        """

        if self.op == 'const':
            self.cval = constants[working_idx]
            return working_idx + 1
        else:
            for s in self.symbols:
                working_idx = s.set_constants(constants, working_idx)
            return working_idx

    def count_constants(self, working_constants):
        """
        Recursively counts the constants in this symbol (including its children).
        """
        if self.op == 'const':
            return working_constants + 1
        for s in self.symbols:
            working_constants += s.count_constants(0)
                
        return working_constants

    def count_symbols(self):
        """
        Recursively counts all symbols that are children of this symbol and updates self.dependents appropriately.
        """
        self.dependents = 0
        for s in self.symbols:
            self.dependents += s.count_symbols() + 1
        return self.dependents

    def mutate(self):
        """
        Randomises the current symbol, adding new random child symbols as appropriate.
        """
        new = Symbol.new_random_symbol(self.n_descriptors, self.depth)
        new.set_random_inputs()
        self.copy(new)

    def get_nodes(self, nodeset=set()):
        """
        recursively constructs a set containing this and all children nodes.
        """
        nodeset.add(self)
        for s in self.symbols:
            s.get_nodes(nodeset)
        return nodeset

    def get_random_symbol(self):
        """
        Get a random symbol from this and its children
        """
        nodes = self.get_nodes()
        return random.choice(list(nodes))

    def mutate_random_node(self):
        node = self.get_random_symbol()
        node.mutate()

    def simplify(self):
        """
        Recursively simplifies this node and all children by looking for constant expressions and evaluating them.
        """

        changed = True
        while changed:
            if self.op in Symbol.zero_order:
                return False
            elif self.op in Symbol.first_order:
                if self.symbols[0].op == 'const':
                    self.cval = self.opdict[self.op]([Symbol('const', 1, 0, cval=self.symbols[0].cval)])
                    self.op = 'const'
                    self.symbols = []
                    self.dependents = 0
                    return True
                else:
                    changed = self.symbols[0].simplify()
            else:
                if self.symbols[0].op == 'const' and self.symbols[1].op == 'const':
                    self.cval = self.opdict[self.op]([Symbol('const', 1, 0, cval=self.symbols[0].cval), Symbol('const', 1, 0, cval=self.symbols[1].cval)])
                    self.op = 'const'
                    self.symbols = []
                    self.dependents = 0
                    return True
                else:
                    changed = self.symbols[0].simplify() or self.symbols[1].simplify()
        self.dependents = self.count_symbols()
        return False

    # 0th order
    def const(self, descriptors: np.array) -> np.array:
        return self.cval
    
    def descriptor(self, descriptors: np.array) -> np.array:
        return descriptors[:, self.descriptor_idx]

    # 1st order
    def sqrt(self, descriptors: np.array) -> np.array:
        return np.sqrt(self.symbols[0].eval(descriptors))

    def pow2(self, descriptors: np.array) -> np.array:
        return self.symbols[0].eval(descriptors)**2

    def pow3(self, descriptors: np.array) -> np.array:
        return self.symbols[0].eval(descriptors)**3

    def bound(self, descriptors: np.array) -> np.array:
        return 1.0/(1.0 + self.symbols[0].eval(descriptors)**2)

    def tanh(self, descriptors: np.array) -> np.array:
        return np.tanh(self.symbols[0].eval(descriptors))

    # 2nd order symbols
    def add(self, descriptors: np.array) -> np.array:
        return self.symbols[0].eval(descriptors) + self.symbols[1].eval(descriptors)

    def subtract(self, descriptors: np.array) -> np.array:
        return self.symbols[0].eval(descriptors) - self.symbols[1].eval(descriptors)

    def multiply(self, descriptors: np.array) -> np.array:
        return self.symbols[0].eval(descriptors) * self.symbols[1].eval(descriptors)

    def divide(self, descriptors: np.array) -> np.array:
        try:
            return self.symbols[0].eval(descriptors) / self.symbols[1].eval(descriptors)
        except ZeroDivisionError:
            return np.nan

    def __str__(self):
        if self.op == 'const':
            return f'{self.cval:.3f}'
        elif self.op == 'descriptor':
            return f'd{self.descriptor_idx}'
        elif self.op == 'sqrt':
            return f'sqrt[{str(self.symbols[0])}]'
        elif self.op == 'pow2':
            return f'[{str(self.symbols[0])}]^2'
        elif self.op == 'pow3':
            return f'[{str(self.symbols[0])}]^3'
        elif self.op == '+':
            return f'({str(self.symbols[0])} + {str(self.symbols[1])})'
        elif self.op == '-':
            return f'({str(self.symbols[0])} - {str(self.symbols[1])})'
        elif self.op == '*':
            return f'{str(self.symbols[0])}*{str(self.symbols[1])}'
        elif self.op == '/':
            return f'{str(self.symbols[0])}/{str(self.symbols[1])}'
        else:
            print(f"ERROR: '{self.op}' is unknown")
            raise ValueError

def optimise_constants(root: Symbol, data: list, targets: list):
    """
    Function to optimise all the constants within a given symbol tree.
    data should be a list of lists of descriptors, e.g.:
    [[1, 2], [2, 3], ...]
    targets should be a list of corresponding oracle values.
    """
    assert data.shape[0] == targets.shape[0], 'Data and targets must be the same length.'

    n = root.count_constants(0)
    if n == 0:
        return []

    bounds = [(-10.0, 10.0) for i in range(n)]
    res = differential_evolution(constant_obj, bounds, args=(root, data, targets), workers=10, polish=False)
    root.set_constants(res.x, 0)
    return res.x

def constant_obj(p: tuple, root: Symbol, data: list, targets: list) -> float:
    """
    Objective function for optimising constants. Expects to be driven by a scipy optimization routine

    p is the tuple of trial constants
    """
    root.set_constants(p, 0)
    return eval_func(root, data, targets)

def eval_func(root: Symbol, data: list, targets: list, length_penalty=0.1) -> float:
    """
    Helper function to evaluate the sum of square errors for a symbol.
    Note: symbol must have constants set before evaluating. 

    length_penalty controls how aggressively we should penalise long expressions
    """
    score = 0

    out = root.eval(data)
    
    if np.any(np.isnan(out)):
        score = 1e30
    else:
        score = (targets - out)**2

    return np.sum(score) + length_penalty*root.dependents

def full_obj(root: Symbol, data: list, targets:list) -> float:
    """
    Objective function for the symbolic optimization
    """
    params = optimise_constants(root, data, targets)
    root.set_constants(params, 0)
    return eval_func(root, data, targets)

def oracle(data):
    return 2*data[:,0]**2 + 3*data[:,1] + 4

def plot_result(root, data, target):
    y = [root.eval(d) for d in data]
    fig, ax = plt.subplots(1)
    ax.plot([d[0] for d in data], target)
    ax.plot([d[0] for d in data], y)
    plt.tight_layout()
    plt.show()

def evolutionary_search(data, targets, maxiter=10, popsize=150, tolerance=0.1, persistence=15):
    """
    Driver to solve the optimisation problem using an evolutionary search.

    maxiter: maximum number of evolutionary epochs allowable.
    popsize: how many candidates should we have in each epoch.
    tolerance: early exit when error is below this threshold.
    persistence: Number of top performing expressions to keep from the previous generation and 
        from which to generate mutants.
    """
    n_descriptors = len(data[0])

    survivors = []

    best = (None, 1e30)

    for i in range(maxiter):
        pop = generate_population(survivors, popsize, n_descriptors)
        results = eval_candidates(pop, data, targets)
        results.sort(key=lambda x: x[1])

        if results[0][1] < tolerance:
            return results[0][0]
        
        print(f"Best new iter {i} was {str(results[0][0])} -> {results[0][1]}")
        if results[0][1] < best[1]:
            best = (results[0][0].clone(), results[0][1])
        print(f"Best in iter {i} was {str(best[0])} -> {best[1]}")

        survivors = [r[0] for r in results[:persistence]]

    return best

def generate_population(survivors: list, popsize: int, n_descriptors: int, proportion_new=0.5):
    """
    Generates a new population of candidate expressions.
    Always keeps survivors, then populates up to popsize with either completely new or mutant 
    expressions as determined by proportion_new.
    """
    to_make = max(0, popsize - len(survivors))
    if len(survivors) == 0:
        n_new = popsize
        n_mutant = 0
    else:
        n_new = int(proportion_new*to_make)
        n_mutant = to_make - n_new

    print(f"Making {to_make}, ({n_new} new, {n_mutant} mutants)")
    out = survivors + [Symbol.new_random_symbol(n_descriptors, 0) for i in range(n_new)]

    for _ in range(n_mutant):
        parent = random.choice(survivors)
        new = parent.clone()
        new.mutate_random_node()
        out.append(new)

    # Now simplify all the expressions to reduce constant count
    for s in out:
        s.simplify()
    return out

def eval_candidates(candidates: list, data: list, targets: list) -> list:
    """
    Wrapper to determine constants and score for each candidate.
    Returns a list of (Symbol, score) tuples.
    """
    out = []
    for c in tqdm(candidates):
        constants = optimise_constants(c, data, targets)
        c.set_constants(constants, 0)
        out.append((c, eval_func(c, data, targets)))
    return out

if __name__ == "__main__":
    # Generate some example descriptor data points
    dx = [x for x in np.linspace(0.01, np.pi, 20)]
    gx, gy = np.meshgrid(dx, dx)
    gx = gx.reshape(gx.shape[0]*gx.shape[1])
    gy = gy.reshape(gy.shape[0]*gy.shape[1])
    data = np.array((gx, gy)).T

    # and the corresponding target values
    targets = oracle(data)

    best = evolutionary_search(data, targets)

    print(str(best[0]), '->', eval_func(best[0], data, targets))

 
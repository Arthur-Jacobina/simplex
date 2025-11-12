# Simplex Algorithm Implementation

A Python implementation of the Simplex algorithm for solving linear programming problems. This implementation uses NumPy for efficient matrix operations and supports both standard and verbose modes for educational purposes.

## Overview

The Simplex algorithm is a widely-used method for solving linear programming problems. This implementation solves **maximization** problems in standard form:

```
Maximize: c^T · x
Subject to: A · x ≤ b
            x ≥ 0
```

Where:
- `c` is the vector of objective function coefficients
- `A` is the constraint matrix
- `b` is the vector of constraint bounds
- `x` is the vector of decision variables

## Requirements

- Python 3.x
- NumPy

Install dependencies:
```bash
uv install
```

## Quick Start

```python
from simplex import Simplex

# Define the problem
c = [300, 250, 450]  # Objective coefficients
A = [[15, 20, 25],   # Constraint matrix
     [35, 60, 60],
     [20, 30, 25],
     [0, 250, 0]]
b = [1200, 3000, 1500, 500]  # Constraint bounds

# Create and solve
simplex = Simplex(c, A, b)
tableau, solution, obj_value = simplex.solve()

# Display results
print(f"Optimal value: {obj_value}")
for var_idx, value in solution:
    print(f"x_{var_idx} = {value}")
```

## API Reference

### `Simplex(c, A, b, verbose=False, max_iters=1000)`

Creates a new Simplex solver instance.

**Parameters:**
- `c` (array-like): Coefficients of the objective function to maximize
- `A` (array-like): Constraint coefficient matrix (m × n)
- `b` (array-like): Right-hand side values of constraints (m × 1)
- `verbose` (bool, optional): If True, prints detailed iteration information. Default: False
- `max_iters` (int, optional): Maximum number of iterations. Default: 1000

**Returns:**
- Simplex instance

### `solve()`

Executes the Simplex algorithm to find the optimal solution.

**Returns:**
- `tableau` (ndarray): Final simplex tableau
- `solution` (list): List of tuples (variable_index, value) for basic variables
- `obj_value` (float): Optimal objective function value

**Special Cases:**
- Returns `(tableau, None, None)` if the problem is unbounded
- Prints warning if maximum iterations are reached

## Examples

### Example 1: Production Planning Problem

Maximize profit from producing three products:

```python
from simplex import Simplex

# Product profits: $300, $250, $450
c = [300, 250, 450]

# Resource constraints:
# - Material: 15, 20, 25 units per product
# - Labor: 35, 60, 60 hours per product
# - Storage: 20, 30, 25 units per product
# - Special constraint: Product 2 minimum
A = [
    [15, 20, 25],    # Material ≤ 1200
    [35, 60, 60],    # Labor ≤ 3000
    [20, 30, 25],    # Storage ≤ 1500
    [0, 250, 0]      # Product 2 ≤ 500
]
b = [1200, 3000, 1500, 500]

simplex = Simplex(c, A, b, verbose=True)
tableau, solution, obj_value = simplex.solve()

print(f"\nOptimal Solution:")
print(f"Maximum Profit: ${obj_value:,.2f}")
for var_idx, value in solution:
    print(f"  Product {var_idx + 1}: {value:.2f} units")
```

### Example 2: Diet Problem

```python
from simplex import Simplex

# Minimize cost (convert to maximization by negating)
# Actually, for minimization you'd need to negate coefficients
# This example shows a direct maximization

# Nutrient values to maximize
c = [4, 3, 2]

# Constraints on ingredients
A = [
    [2, 3, 1],   # Protein ≤ 50
    [1, 1, 2],   # Fat ≤ 30
    [3, 2, 2]    # Carbs ≤ 60
]
b = [50, 30, 60]

simplex = Simplex(c, A, b)
tableau, solution, obj_value = simplex.solve()
```

### Example 3: Using Verbose Mode for Learning

```python
from simplex import Simplex

c = [3, 5]
A = [[1, 0], [0, 2], [3, 2]]
b = [4, 12, 18]

# Enable verbose mode to see each iteration
simplex = Simplex(c, A, b, verbose=True)
tableau, solution, obj_value = simplex.solve()
```

## Algorithm Details

The implementation follows the standard Simplex method:

1. **Initialization**: Constructs the initial tableau from coefficients
2. **Optimality Check**: Checks if all coefficients in objective row are ≤ 0
3. **Pivot Selection**: 
   - **Entering variable**: Column with smallest positive coefficient (Bland's rule variant)
   - **Leaving variable**: Row with minimum ratio test (RHS / pivot column value)
4. **Pivoting**: Performs row operations to make pivot element = 1 and eliminate other column entries
5. **Iteration**: Repeats steps 2-4 until optimal or unbounded

### Key Features

- **Bland's Anti-Cycling Rule**: Uses smallest index tie-breaking to prevent cycling
- **Numerical Stability**: Handles division by zero and floating-point comparisons with tolerances
- **Unboundedness Detection**: Identifies when the problem has no finite optimal solution
- **Educational Mode**: Verbose output shows each iteration for learning purposes

## Contributing

Feel free to open issues and send PRs. Include a brief description of the changes you made.

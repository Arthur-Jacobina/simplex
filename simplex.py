import numpy as np
'''
Example usage: 

   c = [300, 250, 450]
   A = [[15, 20, 25], [35, 60, 60], [20, 30, 25], [0, 250, 0]]
   b = [1200, 3000, 1500, 500]

   simplex = Simplex(c, A, b)
   solution = simplex.solve()
   print(solution)
'''
class Simplex:
    def __init__(self, c, A, b, verbose=False, max_iters=1000):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.max_iters = max_iters
        self.verbose = verbose
        self.tableau = self._build_tableau(self.c, self.A, self.b)
    
    def _build_tableau(self, c, A, b):
        tableau_top = np.column_stack([A, b])
        
        obj_row = np.append(c, 0)
        
        tableau = np.vstack([tableau_top, obj_row])
        
        if self.verbose:
            print("Initial tableau:")
            print(tableau)
        
        return tableau
    
    def _continue(self, tableau):
        obj_row = tableau[-1, :-1]
        return np.any(obj_row > 0)
    
    def _find_pivot(self, tableau):
        obj_row = tableau[-1, :-1]
        if not np.any(obj_row > 0):
            return None, None
        
        positive_indices = np.where(obj_row > 0)[0]
        positive_values = obj_row[positive_indices]
        pivot_col = positive_indices[np.argmin(positive_values)]
        
        rhs = tableau[:-1, -1]
        col = tableau[:-1, pivot_col]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(col > 0, rhs / col, np.inf)
        
        if np.all(ratios == np.inf):
            return pivot_col, None
        
        pivot_row = np.argmin(ratios)
        
        if self.verbose:
            print(f"Pivot: row {pivot_row}, col {pivot_col}")
        
        return pivot_row, pivot_col

    def _pivot(self, tableau):
        pivot_row, pivot_col = self._find_pivot(tableau)
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        
        mask = np.ones(tableau.shape[0], dtype=bool)
        mask[pivot_row] = False
        
        multipliers = tableau[mask, pivot_col:pivot_col+1] 
        tableau[mask] -= multipliers * tableau[pivot_row]
        
        if self.verbose:
            print("Tableau after pivot:")
            print(tableau)
        
        return tableau
        
    def _primal_solution(self, tableau):
        num_vars = tableau.shape[1] - 1
        num_constraints = tableau.shape[0] - 1
        
        solution = []
        
        for j in range(num_vars):
            col = tableau[:, j]
            
            if self._is_pivot_col(col):
                pivot_row = np.argmax(np.abs(col))
                value = tableau[pivot_row, -1]  # RHS value
                solution.append((j, value))
        
        if self.verbose:
            print("Primal solution:")
            for var_idx, val in solution:
                print(f"  x_{var_idx} = {val}")
        
        return solution
    
    def _is_pivot_col(self, col):
        constraint_part = col[:-1]
        
        nonzero_count = np.count_nonzero(np.abs(constraint_part) > 1e-10)
        
        if nonzero_count != 1:
            return False
        
        nonzero_idx = np.argmax(np.abs(constraint_part))
        return np.abs(constraint_part[nonzero_idx] - 1.0) < 1e-10
    
    def solve(self):
        if self.verbose:
            print("Initial tableau:")
            print(self.tableau)
            print()
        
        iteration = 0
        
        while self._continue(self.tableau) and iteration < self.max_iters:
            pivot_row, pivot_col = self._find_pivot(self.tableau)
            
            if pivot_row is None:
                print("Problem is unbounded!")
                return self.tableau, None, None
            
            if self.verbose:
                print(f"Iteration {iteration + 1}")
                print(f"Pivot: row {pivot_row}, col {pivot_col}\n")
            
            self._pivot(self.tableau)
            
            if self.verbose:
                print("Tableau after pivot:")
                print(self.tableau)
                print()
            
            iteration += 1
        
        if iteration >= self.max_iters:
            print("Warning: Maximum iterations reached!")
        
        solution = self._primal_solution(self.tableau)
        obj_value = self._objective_value(self.tableau)
        
        return self.tableau, solution, obj_value
    
    def _objective_value(self, tableau):
        return -tableau[-1, -1]
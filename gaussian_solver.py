import numpy as np
import time
from banded_storage import BandedMatrix

class GaussianEliminationSolver:
    def __init__(self, A, b):
        self.A = A.astype(np.float64).copy()
        self.b = b.astype(np.float64).copy()
        self.n = len(b)
        self.solution = None
        self.solve_time = 0
        
    def solve(self):
        start_time = time.time()
        
        for k in range(self.n - 1):
            for i in range(k + 1, self.n):
                if abs(self.A[k, k]) < 1e-10:
                    raise ValueError(f"Zero pivot encountered at position {k}")
                
                factor = self.A[i, k] / self.A[k, k]
                self.A[i, k:] -= factor * self.A[k, k:]
                self.b[i] -= factor * self.b[k]
        
        self.solution = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            if abs(self.A[i, i]) < 1e-10:
                raise ValueError(f"Zero pivot encountered at position {i}")
            
            self.solution[i] = (self.b[i] - np.dot(self.A[i, i+1:], self.solution[i+1:])) / self.A[i, i]
        
        self.solve_time = time.time() - start_time
        return self.solution
    
    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'dimension': self.n
        }


class BandedGaussianSolver:
    def __init__(self, A, b, p, q):
        self.A = A.astype(np.float64).copy()
        self.b = b.astype(np.float64).copy()
        self.n = len(b)
        self.p = p
        self.q = q
        self.solution = None
        self.solve_time = 0
        
    def solve(self):
        start_time = time.time()
        
        for k in range(self.n - 1):
            if abs(self.A[k, k]) < 1e-10:
                raise ValueError(f"Zero pivot encountered at position {k}")
            
            i_max = min(k + self.p + 1, self.n)
            for i in range(k + 1, i_max):
                factor = self.A[i, k] / self.A[k, k]
                
                j_max = min(min(i + self.q + 1, k + self.q + 1), self.n)
                for j in range(k, j_max):
                    self.A[i, j] -= factor * self.A[k, j]
                
                self.b[i] -= factor * self.b[k]
        
        self.solution = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            if abs(self.A[i, i]) < 1e-10:
                raise ValueError(f"Zero pivot encountered at position {i}")
            
            j_start = i + 1
            j_end = min(i + self.q + 1, self.n)
            
            sum_val = 0
            for j in range(j_start, j_end):
                sum_val += self.A[i, j] * self.solution[j]
            
            self.solution[i] = (self.b[i] - sum_val) / self.A[i, i]
        
        self.solve_time = time.time() - start_time
        return self.solution
    
    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'dimension': self.n,
            'upper_bandwidth': self.q,
            'lower_bandwidth': self.p
        }


class EfficientBandedSolver:
    def __init__(self, A, b):
        if isinstance(A, BandedMatrix):
            self.A = A
            self.n = A.n
            self.p = A.p
            self.q = A.q
        else:
            raise TypeError("A must be a BandedMatrix instance")
        
        self.b = b.astype(np.float64).copy()
        self.solution = None
        self.solve_time = 0
        
    def solve(self):
        start_time = time.time()
        b_work = self.b.copy()
        
        for k in range(self.n - 1):
            pivot = self.A.get(k, k)
            if abs(pivot) < 1e-10:
                raise ValueError(f"Zero pivot encountered at position {k}")
            
            i_max = min(k + self.p + 1, self.n)
            for i in range(k + 1, i_max):
                factor = self.A.get(i, k) / pivot
                
                j_max = min(min(i + self.q + 1, k + self.q + 1), self.n)
                for j in range(k, j_max):
                    old_val = self.A.get(i, j)
                    new_val = old_val - factor * self.A.get(k, j)
                    self.A.set(i, j, new_val)
                
                b_work[i] -= factor * b_work[k]
        
        self.solution = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            pivot = self.A.get(i, i)
            if abs(pivot) < 1e-10:
                raise ValueError(f"Zero pivot encountered at position {i}")
            
            j_start = i + 1
            j_end = min(i + self.q + 1, self.n)
            
            sum_val = 0
            for j in range(j_start, j_end):
                sum_val += self.A.get(i, j) * self.solution[j]
            
            self.solution[i] = (b_work[i] - sum_val) / pivot
        
        self.solve_time = time.time() - start_time
        return self.solution
    
    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'dimension': self.n,
            'upper_bandwidth': self.q,
            'lower_bandwidth': self.p,
            'storage_format': 'banded'
        }

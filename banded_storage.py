import numpy as np

class BandedMatrix:
    def __init__(self, n, p, q):
        self.n = n
        self.p = p
        self.q = q
        self.bandwidth = p + q + 1
        self.data = np.zeros((n, self.bandwidth))
        
    def set(self, i, j, value):
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            return
        
        if abs(i - j) > max(self.p, self.q):
            return
        
        if j < i - self.p or j > i + self.q:
            return
        
        col_index = j - (i - self.p)
        if 0 <= col_index < self.bandwidth:
            self.data[i, col_index] = value
    
    def get(self, i, j):
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            return 0.0
        
        if j < i - self.p or j > i + self.q:
            return 0.0
        
        col_index = j - (i - self.p)
        if 0 <= col_index < self.bandwidth:
            return self.data[i, col_index]
        return 0.0
    
    def get_row_slice(self, i):
        start_col = max(0, i - self.p)
        end_col = min(self.n, i + self.q + 1)
        
        values = []
        for j in range(start_col, end_col):
            values.append(self.get(i, j))
        return np.array(values), start_col, end_col
    
    def to_dense(self):
        dense = np.zeros((self.n, self.n))
        for i in range(self.n):
            for k in range(self.bandwidth):
                j = i - self.p + k
                if 0 <= j < self.n:
                    dense[i, j] = self.data[i, k]
        return dense
    
    def __repr__(self):
        return f"BandedMatrix(n={self.n}, p={self.p}, q={self.q}, storage={self.data.shape})"

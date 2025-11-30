import struct
import numpy as np
from banded_storage import BandedMatrix

class LinearSystemParser:
    def __init__(self, filename, use_banded_storage=False):
        self.filename = filename
        self.use_banded_storage = use_banded_storage
        self.file_id = None
        self.version = None
        self.n = None
        self.q = None
        self.p = None
        self.matrix = None
        self.b = None
        
    def parse_file(self):
        with open(self.filename, 'rb') as f:
            self.file_id, self.version, id1 = struct.unpack('iii', f.read(12))
            
            if self.file_id != 0x0C0A8708:
                raise ValueError(f"Invalid file ID: {hex(self.file_id)}, expected 0x0C0A8708")
            
            if self.version not in [0x102, 0x202]:
                raise ValueError(f"Unsupported version: {hex(self.version)}")
            
            self.n, self.q, self.p = struct.unpack('iii', f.read(12))
            
            if self.version == 0x102:
                self._read_uncompressed_matrix(f)
            elif self.version == 0x202:
                if self.use_banded_storage:
                    self._read_compressed_matrix_banded(f)
                else:
                    self._read_compressed_matrix(f)
            
            self._read_right_hand_side(f)
        
        return self.matrix, self.b
    
    def _read_uncompressed_matrix(self, f):
        total_elements = self.n * self.n
        data = struct.unpack(f'{total_elements}f', f.read(4 * total_elements))
        self.matrix = np.array(data).reshape(self.n, self.n)
    
    def _read_compressed_matrix(self, f):
        self.matrix = np.zeros((self.n, self.n))
        elements_per_row = self.p + self.q + 1
        
        for i in range(self.n):
            row_data = struct.unpack(f'{elements_per_row}f', f.read(4 * elements_per_row))
            
            for k in range(elements_per_row):
                col = i - self.p + k
                if 0 <= col < self.n:
                    self.matrix[i, col] = row_data[k]
    
    def _read_compressed_matrix_banded(self, f):
        self.matrix = BandedMatrix(self.n, self.p, self.q)
        elements_per_row = self.p + self.q + 1
        
        for i in range(self.n):
            row_data = struct.unpack(f'{elements_per_row}f', f.read(4 * elements_per_row))
            self.matrix.data[i, :] = row_data
    
    def _read_right_hand_side(self, f):
        data = struct.unpack(f'{self.n}f', f.read(4 * self.n))
        self.b = np.array(data)
    
    @staticmethod
    def read_header_only(filename):
        with open(filename, 'rb') as f:
            file_id, version, id1 = struct.unpack('iii', f.read(12))
            n, q, p = struct.unpack('iii', f.read(12))
            
            return {
                'file_id': hex(file_id),
                'version': hex(version),
                'version_name': 'Uncompressed' if version == 0x102 else 'Compressed' if version == 0x202 else 'Unknown',
                'n': n,
                'q': q,
                'p': p,
                'bandwidth': p + q + 1
            }
    
    def get_info(self):
        return {
            'file_id': hex(self.file_id) if self.file_id else None,
            'version': hex(self.version) if self.version else None,
            'version_name': 'Uncompressed' if self.version == 0x102 else 'Compressed' if self.version == 0x202 else 'Unknown',
            'n': self.n,
            'q': self.q,
            'p': self.p,
            'bandwidth': self.p + self.q + 1 if self.p is not None and self.q is not None else None
        }

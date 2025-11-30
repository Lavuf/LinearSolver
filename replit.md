# Overview

This is a **Linear System Solver** application designed for solving large-scale sparse linear systems, specifically optimized for strictly diagonally dominant banded matrices. The system is built with Streamlit for the frontend interface and implements Gaussian elimination algorithms with specialized optimizations for banded matrix structures.

The application supports reading binary `.dat` files containing linear system data in both compressed and uncompressed formats, making it particularly suitable for large-scale data applications and deep learning scenarios where sparse linear systems are common.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Decision**: Streamlit-based web interface  
**Rationale**: Provides quick prototyping and deployment of mathematical applications with minimal frontend code. The framework naturally handles real-time updates and interactive controls.

**Key Features**:
- Three operation modes: single file solving, batch processing, and system information
- Sidebar navigation for mode selection
- Real-time performance statistics display
- Wide layout configuration for better visualization of matrices and results

## Backend Architecture

**Core Components**:

1. **File Parser Module (`file_parser.py`)**
   - **Purpose**: Handles reading and parsing binary `.dat` files
   - **Supports**: Two file formats (0x102 uncompressed, 0x202 compressed)
   - **Decision**: Separate parsing logic for compressed vs uncompressed formats
   - **Rationale**: Compressed format only stores banded region elements, requiring different parsing strategies to optimize memory usage

2. **Solver Modules (`gaussian_solver.py`)**
   - **Standard Gaussian Elimination Solver**: For general dense matrices
   - **Banded Gaussian Solver**: Optimized for banded matrices with upper bandwidth (q) and lower bandwidth (p)
   - **Efficient Banded Solver**: Further optimizations for large-scale banded systems
   - **Decision**: Multiple solver implementations
   - **Rationale**: Different matrix structures benefit from different algorithms; banded matrices allow significant performance improvements by only processing non-zero elements within the band

3. **Banded Storage Module (`banded_storage.py`)**
   - **Purpose**: Efficient storage representation for banded matrices
   - **Decision**: Custom storage format storing only bandwidth elements per row
   - **Rationale**: Reduces memory footprint from O(n²) to O(n×bandwidth) for sparse banded matrices
   - **Methods**: get/set operations, row slicing, conversion to dense format

## Data Model

**Binary File Structure**:
1. File identification section (file ID: 0x0C0A8708, version)
2. Matrix information section (dimension n, upper bandwidth q, lower bandwidth p)
3. Coefficient matrix section (float32 values)
4. Right-hand side vector section (float32 values)

**Matrix Representations**:
- Dense numpy arrays for uncompressed format or standard solving
- Custom banded storage for compressed format and optimized solving
- Float64 precision used during computation for numerical stability

## Algorithm Design

**Gaussian Elimination**:
- **Standard approach**: Full elimination on dense matrices
- **Banded optimization**: Only processes elements within the band region (i-p to i+q for row i)
- **Trade-off**: Banded approach has lower computational complexity O(n×bandwidth²) vs O(n³) for dense

**Numerical Considerations**:
- Zero pivot detection with threshold (1e-10)
- Float64 precision for computation despite Float32 storage
- Error handling for singular or near-singular matrices

## Performance Features

**Timing Statistics**:
- Measures solve time for performance analysis
- Tracks dimension and bandwidth for complexity assessment
- Enables comparison between standard and optimized algorithms

**Memory Optimization**:
- Banded storage reduces memory for sparse matrices
- Option to use dense or banded storage based on matrix structure
- Lazy conversion to dense format only when needed

# External Dependencies

## Python Libraries

- **NumPy**: Core numerical computing library for matrix operations and array manipulation
- **Pandas**: Data manipulation and analysis (likely for batch processing results)
- **Matplotlib**: Visualization of matrices, solutions, and performance metrics
- **Streamlit**: Web application framework for the user interface

## File System

- **Input**: Binary `.dat` files with specific format (0x0C0A8708 file ID)
- **Supported versions**: 0x102 (uncompressed) and 0x202 (compressed)
- **No external database**: All data processing is file-based

## Key Design Patterns

- **Strategy Pattern**: Different solver strategies (standard vs banded) selected based on matrix properties
- **Factory Pattern**: Parser creates appropriate matrix representation based on file version and storage preference
- **Template Method**: Common solving workflow with specialized implementations for different matrix types
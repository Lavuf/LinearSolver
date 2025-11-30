import numpy as np
from file_parser import LinearSystemParser
from gaussian_solver import GaussianEliminationSolver, BandedGaussianSolver

def test_file(filename, expected_solution=None):
    print(f"\n{'='*60}")
    print(f"测试文件: {filename}")
    print(f"{'='*60}")
    
    try:
        parser = LinearSystemParser(filename)
        A, b = parser.parse_file()
        info = parser.get_info()
        
        print(f"文件信息:")
        print(f"  - 文件ID: {info['file_id']}")
        print(f"  - 版本: {info['version']} ({info['version_name']})")
        print(f"  - 方程组阶数: {info['n']}")
        print(f"  - 上带宽: {info['q']}")
        print(f"  - 下带宽: {info['p']}")
        print(f"  - 总带宽: {info['bandwidth']}")
        
        print(f"\n矩阵信息:")
        print(f"  - 矩阵形状: {A.shape}")
        print(f"  - 右端向量长度: {len(b)}")
        print(f"  - 矩阵非零元素数: {np.count_nonzero(A)}")
        
        if info['version'] == '0x202' and info['p'] > 0:
            solver = BandedGaussianSolver(A, b, info['p'], info['q'])
            print(f"\n使用带状矩阵优化求解器")
        else:
            solver = GaussianEliminationSolver(A, b)
            print(f"\n使用标准高斯消去法")
        
        solution = solver.solve()
        stats = solver.get_stats()
        
        print(f"\n求解结果:")
        print(f"  - 求解时间: {stats['solve_time']:.6f} 秒")
        print(f"  - 解向量长度: {len(solution)}")
        print(f"  - 解的范围: [{np.min(solution):.6f}, {np.max(solution):.6f}]")
        print(f"  - 解的均值: {np.mean(solution):.6f}")
        print(f"  - 解的标准差: {np.std(solution):.6f}")
        
        if len(solution) <= 20:
            print(f"\n完整解向量:")
            for i, val in enumerate(solution):
                print(f"  x[{i}] = {val:.10f}")
        else:
            print(f"\n前10个解:")
            for i in range(10):
                print(f"  x[{i}] = {solution[i]:.10f}")
            print(f"\n后10个解:")
            for i in range(len(solution)-10, len(solution)):
                print(f"  x[{i}] = {solution[i]:.10f}")
        
        if expected_solution is not None:
            unique_values = np.unique(np.round(solution, 3))
            print(f"\n解的唯一值（保留3位小数）: {unique_values}")
            if len(unique_values) == 1:
                actual = unique_values[0]
                print(f"所有解值均为: {actual}")
                if abs(actual - expected_solution) < 0.001:
                    print(f"✅ 验证通过! 期望值 {expected_solution}, 实际值 {actual}")
                else:
                    print(f"❌ 验证失败! 期望值 {expected_solution}, 实际值 {actual}")
            else:
                print(f"⚠️  解不是单一值，无法直接比较")
        
        residual = np.dot(A, solution) - b
        residual_norm = np.linalg.norm(residual)
        print(f"\n残差验证:")
        print(f"  - 残差范数: {residual_norm:.2e}")
        if residual_norm < 1e-6:
            print(f"  ✅ 残差很小，解是准确的")
        elif residual_norm < 1e-3:
            print(f"  ⚠️  残差较小，解基本准确")
        else:
            print(f"  ❌ 残差较大，解可能不准确")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("线性方程组求解器测试程序")
    print("="*60)
    
    test_files = [
        ("attached_assets/data20251_1764155591805.dat", 1.618, "20阶非压缩"),
        ("attached_assets/data20252_1764155591805.dat", 1.618, "20阶压缩"),
        ("attached_assets/data20253_1764155591805.dat", None, "1500阶非压缩"),
        ("attached_assets/data20254_1764155591805.dat", None, "40000阶压缩"),
    ]
    
    results = []
    for filename, expected, description in test_files:
        print(f"\n\n{'#'*60}")
        print(f"# {description}")
        print(f"{'#'*60}")
        success = test_file(filename, expected)
        results.append((description, success))
    
    print("\n\n" + "="*60)
    print("测试总结")
    print("="*60)
    for desc, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{desc}: {status}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\n总计: {success_count}/{len(results)} 个测试通过")

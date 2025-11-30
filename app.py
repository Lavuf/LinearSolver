import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from file_parser import LinearSystemParser
from gaussian_solver import GaussianEliminationSolver, BandedGaussianSolver, EfficientBandedSolver
from banded_storage import BandedMatrix

st.set_page_config(
    page_title="线性方程组求解器",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 大规模稀疏线性方程组求解系统")
st.markdown("基于高斯消去法的严格对角占优带状矩阵求解器")

st.sidebar.header("操作选项")
mode = st.sidebar.radio(
    "选择模式",
    ["单文件求解", "批量处理", "关于系统"]
)

if mode == "关于系统":
    st.header("系统说明")
    st.markdown("""
    ### 功能特点
    - ✅ 支持二进制.dat格式数据文件读取
    - ✅ 自动识别压缩和非压缩格式
    - ✅ 高斯消去法求解严格对角占优矩阵
    - ✅ 带状矩阵优化算法
    - ✅ 实时性能统计
    
    ### 支持的文件格式
    - **非压缩格式** (0x102): 存储完整的n×n矩阵
    - **压缩格式** (0x202): 仅存储带状区域元素
    
    ### 数据文件结构
    1. **文件标识部分**: 包含文件ID (0x0C0A8708) 和版本号
    2. **矩阵信息部分**: 包含阶数n、上带宽q、下带宽p
    3. **系数矩阵部分**: 矩阵元素(float类型)
    4. **右端常量部分**: 常量向量(float类型)
    
    ### 算法说明
    采用经典的高斯消去法，对于带状矩阵采用优化算法，只处理带状区域内的元素，大幅提升计算效率。
    """)
    
    st.info("💡 提示：本系统特别适用于求解大数据应用和深度学习中的大规模稀疏线性方程组")

elif mode == "单文件求解":
    st.header("单文件求解")
    
    uploaded_file = st.file_uploader("上传.dat数据文件", type=['dat'])
    
    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner("正在解析文件..."):
                info = LinearSystemParser.read_header_only(temp_path)
                use_efficient = info['version'] == '0x202' and info['n'] > 5000
                
                parser = LinearSystemParser(temp_path, use_banded_storage=use_efficient)
                A, b = parser.parse_file()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("方程组阶数", f"{info['n']}")
            with col2:
                st.metric("格式类型", info['version_name'])
            with col3:
                if info['bandwidth']:
                    st.metric("带宽", f"{info['bandwidth']}")
                else:
                    st.metric("带宽", "完整矩阵")
            
            if use_efficient:
                st.info(f"ℹ️ 使用优化的带状矩阵存储格式（内存使用：~{info['n'] * info['bandwidth'] * 4 / 1024 / 1024:.1f} MB，而非 ~{info['n'] * info['n'] * 4 / 1024 / 1024:.1f} MB）")
            
            with st.expander("📊 查看文件详细信息"):
                st.json(info)
            
            if st.button("🚀 开始求解", type="primary"):
                with st.spinner("正在求解方程组..."):
                    try:
                        if isinstance(A, BandedMatrix):
                            solver = EfficientBandedSolver(A, b)
                        elif info['version'] == '0x202' and info['p'] > 0:
                            solver = BandedGaussianSolver(A, b, info['p'], info['q'])
                        else:
                            solver = GaussianEliminationSolver(A, b)
                        
                        solution = solver.solve()
                        stats = solver.get_stats()
                        
                        st.success("✅ 求解成功！")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("求解时间", f"{stats['solve_time']:.6f} 秒")
                        with col2:
                            st.metric("方程组维度", stats['dimension'])
                        
                        st.subheader("📈 解向量")
                        
                        if len(solution) <= 100:
                            df = pd.DataFrame({
                                '索引': range(len(solution)),
                                '解值': solution
                            })
                            st.dataframe(df, use_container_width=True)
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(solution, marker='o', linestyle='-', markersize=3)
                            ax.set_xlabel('Index')
                            ax.set_ylabel('Solution Value')
                            ax.set_title('Solution Vector Distribution')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        else:
                            st.write(f"解向量维度: {len(solution)}")
                            st.write(f"前10个元素: {solution[:10]}")
                            st.write(f"后10个元素: {solution[-10:]}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("最小值", f"{np.min(solution):.6f}")
                            with col2:
                                st.metric("最大值", f"{np.max(solution):.6f}")
                            with col3:
                                st.metric("平均值", f"{np.mean(solution):.6f}")
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            sample_indices = np.linspace(0, len(solution)-1, min(1000, len(solution)), dtype=int)
                            ax1.plot(sample_indices, solution[sample_indices], linestyle='-', linewidth=0.5)
                            ax1.set_xlabel('Index')
                            ax1.set_ylabel('Solution Value')
                            ax1.set_title('Solution Vector Distribution (Sampled)')
                            ax1.grid(True, alpha=0.3)
                            
                            ax2.hist(solution, bins=50, edgecolor='black', alpha=0.7)
                            ax2.set_xlabel('Solution Value')
                            ax2.set_ylabel('Frequency')
                            ax2.set_title('Solution Vector Histogram')
                            ax2.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                        
                        with st.expander("💾 下载解向量"):
                            csv = pd.DataFrame({'solution': solution}).to_csv(index=False)
                            st.download_button(
                                label="下载CSV文件",
                                data=csv,
                                file_name=f"solution_{uploaded_file.name}.csv",
                                mime="text/csv"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ 求解失败: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ 文件解析失败: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

elif mode == "批量处理":
    st.header("批量处理")
    
    data_dir = st.text_input("数据文件目录", value="attached_assets")
    
    if os.path.exists(data_dir):
        dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
        
        if dat_files:
            st.write(f"发现 {len(dat_files)} 个数据文件")
            
            selected_files = st.multiselect(
                "选择要处理的文件",
                dat_files,
                default=dat_files[:4] if len(dat_files) >= 4 else dat_files
            )
            
            if st.button("🚀 批量求解", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, filename in enumerate(selected_files):
                    status_text.text(f"正在处理: {filename}")
                    filepath = os.path.join(data_dir, filename)
                    
                    try:
                        info = LinearSystemParser.read_header_only(filepath)
                        use_efficient = info['version'] == '0x202' and info['n'] > 5000
                        
                        parser = LinearSystemParser(filepath, use_banded_storage=use_efficient)
                        A, b = parser.parse_file()
                        
                        if isinstance(A, BandedMatrix):
                            solver = EfficientBandedSolver(A, b)
                        elif info['version'] == '0x202' and info['p'] > 0:
                            solver = BandedGaussianSolver(A, b, info['p'], info['q'])
                        else:
                            solver = GaussianEliminationSolver(A, b)
                        
                        solution = solver.solve()
                        stats = solver.get_stats()
                        
                        storage_info = " (优化存储)" if use_efficient else ""
                        
                        results.append({
                            '文件名': filename,
                            '阶数': info['n'],
                            '格式': info['version_name'] + storage_info,
                            '带宽': info['bandwidth'] if info['bandwidth'] else 'N/A',
                            '求解时间(秒)': f"{stats['solve_time']:.6f}",
                            '解的范围': f"[{np.min(solution):.4f}, {np.max(solution):.4f}]",
                            '状态': '✅ 成功'
                        })
                        
                    except Exception as e:
                        results.append({
                            '文件名': filename,
                            '阶数': 'N/A',
                            '格式': 'N/A',
                            '带宽': 'N/A',
                            '求解时间(秒)': 'N/A',
                            '解的范围': 'N/A',
                            '状态': f'❌ {str(e)[:50]}'
                        })
                    
                    progress_bar.progress((idx + 1) / len(selected_files))
                
                status_text.text("处理完成！")
                
                st.subheader("📊 批量处理结果")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                success_count = sum(1 for r in results if '✅' in r['状态'])
                st.metric("成功率", f"{success_count}/{len(results)}")
        else:
            st.warning("⚠️ 未找到.dat文件")
    else:
        st.error(f"❌ 目录不存在: {data_dir}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 测试数据说明
- **data20251.dat**: 20阶，非压缩，解=1.618
- **data20252.dat**: 20阶，压缩，解=1.618
- **data20253.dat**: 1500阶，非压缩
- **data20254.dat**: 40000阶，压缩
- **data20255.dat**: 240000阶，压缩（性能测试）
""")

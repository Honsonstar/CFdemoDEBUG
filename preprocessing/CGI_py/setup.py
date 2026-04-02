"""
================================================================================
setup.py - CGI_py 包安装配置
================================================================================

【文件作用】
定义 CGI_py 包的元数据和依赖关系，用于将本目录打包为可安装的 Python 包。

【使用方法】
```bash
# 安装包
pip install .

# 或使用开发模式安装（修改代码后无需重新安装）
pip install -e .

# 完整安装（包含可选依赖）
pip install .[full]
```

【包信息】
- 包名: cgipy
- 版本: 1.0.0
- 描述: CGI (因果图形推理) Python 实现

【依赖】
- numpy >= 1.18.0
- scipy >= 1.5.0

【可选依赖】
- scikit-learn >= 0.22.0 (用于完整功能)

【支持的 Python 版本】
- Python 3.7+

================================================================================
"""

from setuptools import setup, find_packages

setup(
    name='cgipy',
    version='1.0.0',
    description='CGI (Causality Graphical Inference) - Python Implementation',
    author='CGI Authors',
    author_email='',
    url='https://github.com/Causality-Inference/CGI',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.5.0',
    ],
    extras_require={
        'full': [
            'scikit-learn>=0.22.0',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

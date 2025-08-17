from setuptools import setup, find_packages

setup(
    name="retailrocket-recsys",
    version="0.1",
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=[
        'pandas>=2.2.2',
        'numpy>=1.26.4',
        'scikit-learn>=1.5.0',
        'gradio>=4.44.1',
    ],
    python_requires='>=3.10',
)
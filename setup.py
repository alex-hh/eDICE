import setuptools

setuptools.setup(
    name='eDICE',
    version='0.1',
    author = "Alex Hawkins-Hooker, Giovanni Visonà",
    packages=["edice", "edice.data_loaders", "edice.models", "edice.utils"],
    description="epigenomics Data Imputation through Contextualized Embeddings (eDICE)",
    long_description="""
    Software package for the eDICE model presented in the paper "Getting Personal with Epigenetics: Towards Machine-Learning-Assisted Precision Epigenomics" (Hawkins-Hooker et al.)
    """,
    keywords="epigenomics, imputation, transformer, attention, deep learning, machine learning",
    python_requires=">=3.8, <4",
    url = "https://github.com/alex-hh/eDICE",
    license='MIT License',
    install_requires=["tensorflow", "numpy", "pandas", "h5py", "pyYAML", "scikit-learn", "tqdm"],
    # include_package_data=True
)
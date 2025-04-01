from setuptools import setup, find_packages

setup(
    name="NN_Periodogram",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "astropy",
        "tqdm",
        "scipy",
        # Either include direct install from GitHub
        "NN_FAP @ git+https://github.com/username/NN_FAP.git",
        # Or specify it as a dependency if it's on PyPI
        # "NN_FAP",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Flexible Two-Stage NN_FAP Periodogram Analyzer",
    keywords="astronomy, periodogram, time series, NN_FAP",
    url="https://github.com/username/NN_Periodogram",
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'nnp=NNP:main',
        ],
    },
)
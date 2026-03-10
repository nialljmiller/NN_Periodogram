from setuptools import setup

setup(
    name="NN_Periodogram",
    version="0.1.0",
    packages=["NN_Periodogram"],
    package_dir={"NN_Periodogram": "."},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "astropy",
        "tqdm",
        "scipy",
        "NN_FAP @ git+https://github.com/nialljmiller/NN_FAP.git",
    ],
    author="Niall Miller",
    author_email="niall.j.miller@gmail.com",
    description="Flexible Two-Stage NN_FAP Periodogram Analyzer",
    keywords="astronomy, periodogram, time series, NN_FAP",
    url="https://github.com/nialljmiller/NN_Periodogram",
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'nnp=NN_Periodogram.NNP:main',
        ],
    },
)
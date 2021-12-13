from setuptools import find_packages, setup
import versioneer

setup(
    name="xeof",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Dougie Squire",
    url="https://github.com/dougiesquire/xeof",
    description="A simple dask-enabled xarray wrapper for empirical orthogonal function decomposition",
    long_description="A simple dask-enabled xarray wrapper for empirical orthogonal function decomposition using svd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "xarray",
        "dask",
    ],
)

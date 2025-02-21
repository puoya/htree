from setuptools import setup, find_packages
from .__version__ import __version__

# Read the version from the package's __version__.py file
version = {}
with open(os.path.join("your_package", "__version__.py")) as fp:
    exec(fp.read(), version)


setup(
    name='your_package',
    version=version['__version__'],
    description='A library for tree reading, embedding, and analysis',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/your_package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
from setuptools import setup, find_packages

setup(
    name="htree",
    version="3.4.0",
    description="A library for tree reading, embedding, and analysis of phylogenetic trees",
    author="Puoya Tabaghi",
    author_email="ptabaghi@ucsd.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "torch",
        "treeswift",
        "tqdm",
        "imageio",
        "imageio-ffmpeg"
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10.16",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # Remove the `license_files` line, as it's causing issues
    # license_files=["LICENSE"],  # Remove this line
)

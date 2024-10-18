from setuptools import setup, find_packages

setup(
    name='WHOMP',  # Your package name
    version='0.1.0',  # Initial release version
    author='Shizhou XU',  # Your name
    author_email='shzxu@ucdavis.edu',  # Your email
    description='Implementation of algorithms proposed in WHOMP: Optimizing Randomized Controlled Trials via Wasserstein Homogeneity',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/xushizhou/WHOMP',  # URL to your project (e.g., GitHub repo)
    packages=find_packages(),  # Automatically find your package(s)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
    install_requires=[
        # Add any dependencies your package needs here, e.g., 'numpy', 'scipy'
        'numpy', 'scipy', 'sklearn', 'joblib', 'random', 'ot'
    ],
)
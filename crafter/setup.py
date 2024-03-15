import setuptools
import pathlib


setuptools.setup(
    name='crafter',
    version='1.8.2',
    description='Open world survival game for reinforcement learning.',
    url='http://github.com/danijar/crafter',
    packages=['crafter'],
    package_data={'crafter': ['data.yaml', 'assets/*']},
    install_requires=[
        'numpy', 'imageio', 'pillow', 'opensimplex', 'ruamel.yaml',
    ],
    extras_require={'gui': ['pygame']},
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

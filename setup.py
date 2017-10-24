from setuptools import setup, find_packages

setup(
    name='nengo_conductance_synapses',
    packages=[
        'nengo_conductance_synapses', 'nengo_conductance_synapses.sim_cond_exp'
    ],
    package_data={
        'nengo_conductance_synapses.sim_cond_exp':
        ['Makefile', 'sim_cond_exp_pipe.cpp', 'json.hpp']
    },
    version='0.1',
    author='Andreas Stöckel',
    description='Conductance based synapses in Nengo',
    url='https://github.com/ctn-waterloo/nengo_conductance_synapses',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ])


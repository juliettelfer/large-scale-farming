import _functions_.plotting as plot
import os

'''
This file is used for running the plotting functions.
Begin by defining your variables then choose a function
to run from _functions_/plotting.py
'''
sources = ["CoC"]

plot.variables_scatter_plot(sources, 'Yield (kg)', 
                            'impacts.gwp100.value',
                            'riceGrainInHuskFlooded')

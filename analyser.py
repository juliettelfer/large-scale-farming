import _functions_.plotting as plot
import os

'''
This file is used for running the plotting functions.
Begin by defining your variables then choose a function
to run from _functions_/plotting.py
'''
# Sources inputted into a plotting function must be a list

sources = os.listdir("sources/")

plot.indiactor_barchart(["LCAS"], "co2EqGwp100Ipcc2021", 
                        "GWP100 (kgCO2Eq, IPCC 2021)")

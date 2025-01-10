import _functions_.processing as request
import os

source = 'LCAS'
product = "riceGrainInHuskFlooded"
countries = ['Bangladesh', 'India', 'Nepal']

for country in countries:

    request.get_indicator_contributions(product, country, source, 
                                        "co2EqGwp100Ipcc2021")
import pandas as pd
import numpy as np
import os
# import itertools


def get_dataframes(self, country, source):

    '''
    Returns dataframes for cycle, impact, and site.
    self: the product term.id
    country: the country name
    source: the source folder
    '''
    cycle = pd.read_csv(f"sources/{source}/{country}/{self}/"
                        "cycle.csv")
    impact = pd.read_csv(f"sources/{source}/{country}/{self}/"
                         "impactAssessment.csv")
    site = pd.read_csv(f"sources/{source}/{country}/{self}/site.csv")
    
    return cycle, impact, site


def get_products(self, country, source):

    '''
    Returns arrays of the products in a cycle, split into primary products
    and crop residues
    self: the product term.id
    country: the country name
    source: the source folder
    '''

    save_directory = f"arrays/{source}/{country}/{self}/cycle/product/"
    os.makedirs(save_directory, exist_ok=True)  # noqa: F821

    # Filter cycle for primary product
    cycle = get_dataframes(self, country, source)[0]
    primary_product = cycle.loc[:, (cycle.columns.str.startswith(
        f"products.{self}"))]
    primary_product.columns = primary_product.columns.str.replace(
        f"products.{self}.", "")

    # Normalise yield depending on mean and median
    primary_product['normalisedYieldMean'] = primary_product[
                            "value"].divide(primary_product["value"].mean())
    primary_product['normalisedYieldMedian'] = primary_product[ 
                            "value"].divide(primary_product["value"].median())

    # Save as numpy array
    primary_product_array = primary_product.to_records(index=False)
    np.save(f"{save_directory}primary_product", primary_product_array)

    # Filter for the crop residue and yield
    # residue_product = cycle.loc[:, 
    # (cycle.columns.str.contains("CropResidue"))
    #                             | (cycle.columns.str.startswith(
    #                                 f"products.{self}.value"))]
    # residue_product.columns = residue_product.columns.str.replace(
    #     "products.", "")
    
    # Add total residue columns normalised to the average yield
    # residue_product['aboveGroundCropResidueTotal.relativeToYield.value'
    #                 ] = residue_product["aboveGroundCropResidueTotal.value"
    #                                     ].divide(residue_product[
    #                                         f"{self}.value"].mean())
    # residue_product['belowGroundCropResidue.relativeToYield.value'
    #                 ] = residue_product["belowGroundCropResidue.value"
    #                                     ].divide(residue_product[
    #                                         f"{self}.value"].mean())

    # Save as array
    # residue_product_array = residue_product.to_records(index=False)
    # np.save(f"{save_directory}crop_residue", residue_product_array)


def get_inputs(self, country, source):

    '''
    Returns arrays of inputs into the cycle. The raw values are
    saved for all inputs as well as values normalised wrt to the mean
    and median value.
    self: the product term.id
    country: the country name
    source: the source folder
    '''
    save_directory = f"arrays/{source}/{country}/{self}/cycle/input/"
    os.makedirs(save_directory, exist_ok=True)  # noqa: F821

    # Input arrays
    cycle = get_dataframes(self, country, source)[0]
    cycle = cycle.loc[:, (cycle.columns.str.endswith('.value'))]
    inputs = cycle.loc[:, (cycle.columns.str.startswith('inputs'))]
    inputs.columns = inputs.columns.str.replace('inputs.', '')
    inputs.columns = inputs.columns.str.replace('.value', '')
    
    # Save all the inputs as an array
    inputs_array = inputs.to_records(index=False)
    np.save(f"{save_directory}all_inputs", inputs_array)
    
    # Normalise wrt to the mean and save separately
    normalised_inputs = pd.DataFrame()
    for column in inputs:
        normalised_inputs[column] = inputs[column].divide(inputs[
            column].mean())

    normalised_inputs_array = normalised_inputs.to_records(index=False)
    np.save(f"{save_directory}normalised_all_inputs", normalised_inputs_array)


def get_variables_impacts(self, country, source):

    '''
    Returns arrays of variables that we want to observe, these are yield, 
    grouped fertiliser inputs, fuel inputs (including electricity), operations. 
    It also contains the impact indicators and endpoints that we want to 
    observe, making sure the right cycles are correlated.  
    self: the product term.id.
    country: the country name
    source: the source folder
    '''
    save_directory = f"arrays/{source}/{country}/{self}/cycle/"
    os.makedirs(save_directory, exist_ok=True)  # noqa: F821

    # Make empty dataframe
    variables = pd.DataFrame()

    # Add the cycle ids
    cycle = get_dataframes(self, country, source)[0]
    variables['cycle.id'] = cycle['@id']

    # Get yield

    # Filter cycle for primary product
    cycle = get_dataframes(self, country, source)[0]
    primary_product = cycle.loc[:, (cycle.columns.str.startswith(
        f"products.{self}"))]
    primary_product.columns = primary_product.columns.str.replace(
        f"products.{self}.", "")
    
    variables['Yield (kg)'] = primary_product['value']

    # Get operations variables

    cycle = get_dataframes(self, country, source)[0]
    cycle = cycle.loc[:, (cycle.columns.str.endswith('.value'))]
    practice = cycle.loc[:, (cycle.columns.str.startswith('practices'))]
    practice.columns = practice.columns.str.replace('practices.', '')
    practice.columns = practice.columns.str.replace('.value', '')

    practice.columns = practice.columns.str.split('+')
    practice = practice.groupby([
        c[0] for c in practice.columns], axis=1).sum()

    operation_ref = pd.read_csv("reference/operation.csv")
    operation = practice.loc[:, (practice.columns.isin(
        operation_ref['term.id']))]
    
    # Create a total irrrigation column
    irrigating = operation.loc[:, (operation.columns.str.startswith(
        "irrigating"))]
    irrigating['totalIrrigating'] = irrigating.sum(axis=1)

    # Create a total machinery use column
    machinery_use = operation.copy()
    for index, column in enumerate(machinery_use):
        if column == "machineryUseOperationUnspecified":
            machinery_use[f"machinery.{column}"] = machinery_use[column]
        else:
            fuel_use = str(operation_ref.loc[operation_ref[
                'term.id'] == column, 'lookups.1.dataState'].item())
            if fuel_use == 'complete':
                machinery_use[f"machinery.{column}"] = machinery_use[column]

    machinery_use['totalMachineryUse'] = machinery_use.loc[:, (
        machinery_use.columns.str.startswith("machinery."))].sum(axis=1)

    # Add totals to variables
    variables['Irrigation (hours)'] = irrigating['totalIrrigating']
    variables['Machinery (hours)'] = machinery_use['totalMachineryUse']

    # Get fertiliser variables

    # Input arrays
    cycle = get_dataframes(self, country, source)[0]
    cycle = cycle.loc[:, (cycle.columns.str.endswith('.value'))]
    inputs = cycle.loc[:, (cycle.columns.str.startswith('inputs'))]
    inputs.columns = inputs.columns.str.replace('inputs.', '')
    inputs.columns = inputs.columns.str.replace('.value', '')

    # Load fertiliser glossaries and filter inputs
    inorganic_fertiliser_ref = pd.read_csv(
        "reference/fertiliser/inorganicFertiliser.csv")
    organic_fertiliser_ref = pd.read_csv(
        "reference/fertiliser/organicFertiliser.csv")
    brand_name_fertiliser_ref = pd.read_csv(
        "reference/fertiliser/fertiliserBrandName.csv")
    
    fertiliser = inputs.loc[:, (inputs.columns.isin(
        inorganic_fertiliser_ref['term.id']))]
    organic_fertiliser = inputs.loc[:, (inputs.columns.isin(
        organic_fertiliser_ref['term.id']))]
    brand_name_fertiliser = inputs.loc[:, (inputs.columns.isin(
        brand_name_fertiliser_ref['term.id']))]
    
    # Add inorganic fertiliser column
    variables['Inorganic Fertiliser (kg)'] = fertiliser.sum(axis=1)
    
    # Add columns for N, P205 and K2)
    for index, column in enumerate(organic_fertiliser):
        if column.endswith("KgMass"):
            factor_N_value = float(organic_fertiliser_ref.loc[
                organic_fertiliser_ref['term.id'] == column,
                'term.defaultProperties.1.value'])
            fertiliser[f"{column}KgN"] = organic_fertiliser[
                column].mul(factor_N_value/100)
            factor_P_value = float(organic_fertiliser_ref.loc[
                organic_fertiliser_ref['term.id'] == column,
                'term.defaultProperties.3.value'])
            fertiliser[f"{column}KgP2O5"] = organic_fertiliser[
                column].mul(factor_P_value/100)
            factor_K_value = float(organic_fertiliser_ref.loc[
                organic_fertiliser_ref['term.id'] == column,
                'term.defaultProperties.4.value'])
            fertiliser[f"{column}KgK2O"] = organic_fertiliser[
                column].mul(factor_K_value/100)
   
    for index, column in enumerate(brand_name_fertiliser):
        factor_N_value = float(brand_name_fertiliser_ref.loc[
            brand_name_fertiliser_ref['term.id'] == column,
            'lookups.0.value'])
        fertiliser[f"{column}KgN"] = brand_name_fertiliser[
            column].mul(factor_N_value/100)
        factor_P_value = float(brand_name_fertiliser_ref.loc[
            brand_name_fertiliser_ref['term.id'] == column,
            'lookups.1.value'])
        fertiliser[f"{column}KgP2O5"] = brand_name_fertiliser[
            column].mul(factor_P_value/100)
        factor_K_value = float(brand_name_fertiliser_ref.loc[
            brand_name_fertiliser_ref['term.id'] == column,
            'lookups.2.value'])
        fertiliser[f"{column}KgK2O"] = brand_name_fertiliser[
            column].mul(factor_K_value/100)
        
    # Group by N, P2O5, K2O
    grouped_fertiliser = fertiliser.copy()
    
    if grouped_fertiliser.empty is False: 

        grouped_fertiliser.columns = grouped_fertiliser.columns.str.split('Kg')
        grouped_fertiliser = grouped_fertiliser.groupby([
            c[-1] for c in grouped_fertiliser.columns], axis=1).sum()
        grouped_fertiliser.columns = grouped_fertiliser.columns.str.split(
            '.')
        grouped_fertiliser = grouped_fertiliser.groupby([
            c[0] for c in grouped_fertiliser.columns], axis=1).sum()
    print(grouped_fertiliser)

    if 'N' in grouped_fertiliser:
        variables['N Fertiliser (kg)'] = grouped_fertiliser['N']
    else:
        variables['N Fertiliser (kg)'] = np.nan
    if 'P2O5' in grouped_fertiliser:
        variables['P2O5 Fertiliser (kg)'] = grouped_fertiliser['P2O5']
    else:
        variables['P2O5 Fertiliser (kg)'] = np.nan
    if 'K2O' in grouped_fertiliser:
        variables['K2O Fertiliser (kg)'] = grouped_fertiliser['K2O']
    else:
        variables['K2O Fertiliser (kg)'] = np.nan

    # Add total fertiliser column
    variables['Total Fertiliser'] = variables['N Fertiliser (kg)'] + variables[
        'P2O5 Fertiliser (kg)'] + variables['K2O Fertiliser (kg)']
    
    # Get fuel and electricity variables

    # Load reference csvs and filter inputs
    fuel_ref = pd.read_csv("reference/fuel.csv")
    electricity_ref = pd.read_csv("reference/electricity.csv")
    
    fuel = inputs.loc[:, (inputs.columns.isin(
        fuel_ref['term.id']))]
    electricity = inputs.loc[:, (inputs.columns.isin(
        electricity_ref['term.id']))]
    
    variables['Fuel (kg)'] = fuel.sum(axis=1)
    variables['Electricity (kWh)'] = electricity.sum(axis=1)

    # Get impacts and endpoints

    # Get the indicator/endpoint array
    impact = get_dataframes(self, country, source)[1]
    impacts = impact.loc[:, (impact.columns.str.endswith('.value'))]
    impacts = impacts.loc[:, (impacts.columns.str.startswith(
        'endpoints') | impacts.columns.str.startswith('impacts'))]
    impacts['cycle.id'] = impact['cycle.@id']

    # Add the impacts columns to variables
    variables = variables.merge(impacts)
    
    # Save as csv (easier to debug)

    variables.to_csv(f"{save_directory}variables.csv")


def get_lat_long_site(self, country, source):

    '''
    Returns an array of the lattiudes and longitudes of sites assocaited 
    with a primary product.
    self: the product term.id
    country: the country name
    source: the source folder
    '''
        
    # Create save directory
    save_directory = f"arrays/{source}/{country}/{self}/site/"
    os.makedirs(save_directory, exist_ok=True)

    # Get site and region CSV
    site = get_dataframes(self, country, source)[2]
    region = pd.read_csv("reference/region.csv")
    
    # Rename region column to fit with site
    site = site.rename(columns={"region.@id": "term.id"})
    
    # Merge only adding latitude and longitude columns
    site = pd.merge(site, region[['term.id', 'term.latitude', 'term.longitude']
                                 ], on='term.id', how='left')

    # Save as an array
    site_array = site.to_records(index=False)
    np.save(f"{save_directory}lat_long", site_array)


def get_indicator_contributions(self, country, source, indicator):

    '''
    Returns array of the impacts multiplited by the conversion factors for an
    inidicator, e.g. co2EqGwp100Ipcc2021.
    self: the product term.id
    country: the country name
    source: the source folder
    indicator: the inidicator
    '''

    # Save directory for numpy arrays
    save_directory = f"arrays/{source}/{country}/{self}/impacts/{indicator}"
    os.makedirs(save_directory, exist_ok=True)

    # Get impact and factor dataframes
    impact = get_dataframes(self, country, source)[1]
    factor_ref = pd.read_csv("reference/emission.csv")

    # Filter impacts for term names with 'emission' and 'value'
    emissions = impact.loc[:, (impact.columns.str.startswith(
        'emission') & impact.columns.str.endswith('.value'))]

    # Mulitplying each emission by the corresponding factor
    for i, factor in enumerate(factor_ref):
        
        factor_value = factor_ref.loc[i, str(indicator)]
        emissions.loc[:, (emissions.filter(like=factor).columns)
                      ] = emissions.loc[:, (emissions.filter(
                          like=factor).columns)].mul(factor_value)
        
    # Group emission, drop columns where all values are 0
    group_emissions = emissions.copy()
    group_emissions.columns = group_emissions.columns.str.split('[+/.]')
    group_emissions = group_emissions.groupby([
        c[1] for c in group_emissions.columns], axis=1).sum()
    group_emissions = group_emissions.loc[:, group_emissions.any()]
    
    # Group input emissions, drop columns where all values are 0
    input_emissions = emissions.filter(like='InputsProduction')
    group_input_emissions = input_emissions.copy()
    print(group_input_emissions.columns.values)
    group_input_emissions.columns = group_input_emissions.columns.str.split(
        'inputs[\[]')
    group_input_emissions = group_input_emissions.groupby(
        [c[0] for c in group_input_emissions.columns], axis=1).sum()
    
    group_input_emissions.columns = group_input_emissions.columns.str.split(
         '[\]]')
    group_input_emissions = group_input_emissions.groupby(
         [c[0] for c in group_input_emissions.columns], axis=1).sum()
    
    # Remove empty columns
    grouped_input_emissions = group_input_emissions.loc[
        :, group_input_emissions.any()]
  
    # Remove emissions that do not have factors in the csv - this is here for
    # the land occupation emissions that were occurring

    factor_list = factor_ref.loc[:, 'term.id']
    group_emissions_list = group_emissions.columns.values.tolist()
    unwanted_list = list(set(group_emissions_list) - set(factor_list))

    if unwanted_list:
        pattern = '|'.join(unwanted_list)
        group_emissions = group_emissions.drop(list(
            group_emissions.filter(regex=pattern)), axis=1)

    # Save arrays
    group_emissions_array = group_emissions.to_records(index=False)
    np.save(save_directory + "/group_emissions", group_emissions_array)

    group_input_emissions_array = grouped_input_emissions.to_records(
        index=False)
    np.save(save_directory + "/group_input_emissions", 
            group_input_emissions_array)

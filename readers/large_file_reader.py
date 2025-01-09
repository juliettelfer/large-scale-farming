import pandas as pd
import numpy as np
import os

'''
This file will read a large CSV and split it into smaller CSVs, based on the
country and theprimary product term.id. These will be stored inside your source
folder in /sources, with the sub-folders being the country, then the product
term.id.

'''
# sources = os.listdir("sources")
sources = ['LCAS']

for source in sources:
    
    # Retreive the filepaths for the cycle, impact and site
    cycle = f"sources/{source}/Cycle-recalculated-compacted.csv"
    impact = f"sources/{source}/ImpactAssessment-recalculated-compacted.csv"
    site = f"sources/{source}/Site-recalculated-compacted.csv"

    # Retrieve a list of the unique country names from the site csv
    countries = pd.read_csv(site, usecols=["country.name", "@id"], 
                            low_memory=False)
    unique_countries = list(set(countries["country.name"]))
    
    for i, country in enumerate(unique_countries):

        # Retrieve a list of site ids in that country
        site_ids = countries[countries["country.name"] == country]["@id"]

        # Find the impact and cycles indexes that match the site ids
        impact_ids = pd.read_csv(impact, usecols=['@id', 'site.@id'], 
                                 low_memory=False)
        country_impact_index = np.array(impact_ids.index[impact_ids[
            'site.@id'].isin(site_ids)])
        country_impact_index = list(country_impact_index + 1) + [0]
        
        # Retrieve a list of the unique primary product ids from the impact csv
        products = pd.read_csv(impact, usecols=[
            "product.term.@id", "product.primary", "@id"], 
            skiprows=lambda x: x not in country_impact_index,
            low_memory=False)
    
        primary_products = list(set(products[products[
            'product.primary'] == True]['product.term.@id']))  # noqa E712

        for index, id in enumerate(primary_products):

            # Only read impact rows containing primary product
            primary_product_ids = products[products['product.term.@id'] == id]
            primary_product_indexes = list(primary_product_ids.index + 1) + [0]
            impact_df = pd.read_csv(
                impact, 
                skiprows=lambda x: x not in primary_product_indexes)

            # Filter cycle for matching cycle.@id
            cycle_ids = pd.read_csv(cycle, usecols=["@id"],
                                    low_memory=False)
            impact_cycle_ids = impact_df['cycle.@id'].tolist()
            matchine_cycle_ids = cycle_ids[cycle_ids["@id"].isin(
                impact_cycle_ids)]
            matching_cycle_indexes = list(matchine_cycle_ids.index + 1) + [0]
            cycle_df = pd.read_csv(
                cycle, skiprows=lambda x: x not in matching_cycle_indexes,
                low_memory=False)
            
            # Filter site for matching site.@id
            cycle_site_ids = cycle_df['site.@id'].tolist()
            matching_site_ids = site_ids[site_ids.isin(
                cycle_site_ids)]
            matching_site_indexes = list(matching_site_ids.index + 1) + [0]
            site_df = pd.read_csv(
                site, skiprows=lambda x: x not in matching_site_indexes, 
                low_memory=False)
            
            # Delete unnecessary columns
            cycle_df = cycle_df.loc[:, ~(
                cycle_df.columns.str.startswith('emissions.'))]
            cycle_df = cycle_df.loc[:, ~(
                cycle_df.columns.str.contains('methodClassification'))]
            cycle_df = cycle_df.loc[:, ~(
                cycle_df.columns.str.endswith('odel.@id'))]
            
            site_df = site_df.loc[:, ~(
                site_df.columns.str.endswith('endDate'))]
            site_df = site_df.loc[:, ~(
                site_df.columns.str.endswith('source.@id'))]

            # Remove columns with only blanks, then fill other blanks with 0
            cycle_df = cycle_df.loc[:, cycle_df.any()]
            impact_df = impact_df.loc[:, impact_df.any()]
            site_df = site_df.loc[:, site_df.any()]

            cycle_df = cycle_df.fillna(0)
            impact_df = impact_df.fillna(0)
            site_df = site_df.fillna(0)

            # Save CSVs in folders depending on source and product name
            save_directory = f"sources/{source}/{country}/{id}"
            os.makedirs(save_directory, exist_ok=True)

            cycle_df.to_csv(f"{save_directory}/cycle.csv")
            impact_df.to_csv(f"{save_directory}/impactAssessment.csv")
            site_df.to_csv(f"{save_directory}/site.csv")

 
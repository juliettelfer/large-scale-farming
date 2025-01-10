import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import _functions_.processing as request
import itertools
import cartopy.crs as ccrs
import cartopy.feature as cf
from math import log10, floor
from itertools import chain
# import re


def get_sub_directories(directory):
    lowest_dirs = list()
    countries = list()
    products = list()
    for root, dirs, files in os.walk(directory):
        if not dirs:
            root = os.path.relpath(root, "sources/").replace("\\", "/")
            lowest_dirs.append(root)
            products.append(os.path.split(root)[1])
            countries.append(os.path.split(os.path.split(root)[0])[1])
    return lowest_dirs, products, countries


def yield_boxplot(sources):

    '''
    Returns a boxplot of the primary product yields, normalised to the average
    yield. The figure also contains a plot of the difference between FAOSTAT
    yield and the mean yield.
    sources: list of source folders
    '''

    # Save directory for plots
    plot_save_directory = "plots/cycle/products/"
    os.makedirs(plot_save_directory, exist_ok=True)

    for source in sources:

        # Retrive paths, products and countries
        path_list, products, countries = get_sub_directories(
            f"sources/{source}/")

        # Create figure 
        count = len(products)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                 gridspec_kw={'height_ratios': [1, 2]})
    
        # Import FAOSTAT yields
        fao_yield_df = pd.read_csv("reference/FAOSTAT_yields.csv")

        # Create emptys lists
        difference_list = []
        mean_list = []
        fao_list = []
        label_list = []

        for i, path in enumerate(path_list):

            # Run processing code if array directory doesn't exist
            product_save_directory = f"arrays/{path}/cycle/product/primary_product.npy" ## noqa
            if not os.path.exists(product_save_directory):
                request.get_products(products[i], countries[i], source)
                
            # Load arrays and convert back to dataframes
            primary_product_array = np.load(product_save_directory, 
                                            allow_pickle=True)
            primary_product = pd.DataFrame.from_records(
                primary_product_array)

            # Boxplot normalised yield
            axes[1].boxplot(primary_product["normalisedYieldMean"], 
                            showfliers=True, positions=[i], widths=[
                                0.75])

            # Retrieve yields, calculate difference and append to list
            mean_yield = float(primary_product["value"].mean())
            fao_yield = float(fao_yield_df[(fao_yield_df[
                'product'] == products[i]) & (fao_yield_df[
                    'country'] == countries[i])]["yield"].iloc[0])
            mean_list.append(int(mean_yield))
        
            if fao_yield == fao_yield:
                percent_difference = ((mean_yield - fao_yield)/((
                    fao_yield + mean_yield)/2))*100
                difference_list.append(percent_difference)
                fao_list.append(int(fao_yield))
            else:
                percent_difference = 0
                difference_list.append(percent_difference)
                fao_list.append(np.nan)
            
            # Retrieve the label from the FAO file
            fao_label = fao_yield_df[(fao_yield_df[
                'product'] == products[i]) & (fao_yield_df[
                    'country'] == countries[i])]["label"].iloc[0]
            label_list.append(fao_label)

            # Plot bar of percentage difference
            axes[0].bar(i, percent_difference, align='center', width=0.75,
                        color='darkGrey')
            
        axes[0].set_title(f"Yield per Ha, Difference between {source} Mean"
                          " and FAOSTAT", fontsize=14)
        axes[0].set_ylabel("Percentage %", fontsize=12)

        # Add the values of mean and FAO yield as text
        rects = axes[0].patches
        for rect, mean, fao in zip(rects, mean_list, fao_list):
            axes[0].text(rect.get_x() + rect.get_width() / 2, -count - 3,
                         mean, ha="center", va="bottom", color='darkred')
            axes[0].text(rect.get_x() + rect.get_width() / 2, -count - 9, 
                         fao, ha="center", va="bottom", color='darkblue')

        axes[0].text(-0.5, -count - 3, "Mean (kg)", ha="center", 
                     va="bottom", color='darkred')
        axes[0].text(-0.5, -count - 9, "FAO (kg)", ha="center", 
                     va="bottom", color='darkblue')
        axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
        axes[0].set_xlim(-0.5, count - 0.5)
        axes[1].set_xlim(-0.5, count - 0.5)

        # Figure alterations  
        plt.xticks(rotation=90)

        axes[1].set_ylabel("Normalised yield", fontsize=12)
        axes[1].tick_params(axis='x', labelrotation=90)
        axes[1].set_xticklabels(label_list, fontsize=12)
        axes[1].set_xlabel("Products", fontsize=12)

        axes[1].set_title("Yield per Ha, Normalised to the Mean", 
                          fontsize=14)

        axes[1].grid(True)
        axes[0].grid(True)

    fig.tight_layout()
    fig.savefig(
        f"{plot_save_directory}{source}_yield_boxplot",
        bbox_inches='tight')


def fertiliser_operation_barchart(sources):

    '''
    Returns a barchat with the fertiliser inputted, and the operating
    hours for machinery and irrigation.
    sources: list of source folders
    '''

    # Save directory for plots
    plot_save_directory = "plots/cycle/"
    os.makedirs(plot_save_directory, exist_ok=True)

    # Load FAO csv for the labels
    fao_yield_df = pd.read_csv("reference/FAOSTAT_yields.csv")
    
    for source in sources:

        # Retrive paths, products and countries
        path_list, products, countries = get_sub_directories(
            f"sources/{source}/")

        # Create empty dataframe for averages
        average_fertiliser_df = pd.DataFrame()
        average_operation_df = pd.DataFrame()

        for i, path in enumerate(path_list):

            # Retrieve variable csv
            variables_save_directory = f"arrays/{path}/cycle/variables.csv" ## noqa

            if not os.path.exists(variables_save_directory):
                request.get_variables_impacts(products[i], countries[i], 
                                              source)
            
            variables = pd.read_csv(variables_save_directory)

            # Retrieve the labels from the FAO csv

            fao_label = fao_yield_df[(fao_yield_df[
                'product'] == products[i]) & (fao_yield_df[
                    'country'] == countries[i])]["label"].iloc[0]
    
            # Append averages to dataframes
            average_fertiliser = [variables['N Fertiliser (kg)'].mean(), 
                                  variables['P2O5 Fertiliser (kg)'].mean(), 
                                  variables['K2O Fertiliser (kg)'].mean()]
            average_fertiliser_df[fao_label] = pd.Series(
                average_fertiliser,
                index=[
                    'N fertiliser', 'P2O5 Fertiliser', 'K20 Fertiliser'])
    
            average_operation = [variables['Irrigation (hours)'].mean(),
                                 variables['Machinery (hours)'].mean()]
            average_operation_df[fao_label] = pd.Series(
                average_operation,
                index=['Total Irrigation', 'Total Machinery'])
      
        average_fertiliser_df = average_fertiliser_df.transpose()
        average_operation_df = average_operation_df.transpose()

        # Plot values on a stacked bar chart
        colour_list = ['darkred', 'darkorange', 'gold', 'green', 
                       'turquoise']
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))
        axes2 = axes.twinx()

        average_fertiliser_df.plot(kind='bar', stacked=True, ax=axes, 
                                   color=colour_list, position=1, 
                                   width=0.35)
        axes.legend(loc="upper left")

        average_operation_df.plot(kind='bar', stacked=True, ax=axes2,
                                  color=["lightblue", "green"],
                                  position=0, width=0.35)
        axes2.legend(labels=["Irrigation", "Machinery"], loc="upper right")
        
        # Polish up plot
        axes.set_ylabel("Mass (kg)", fontsize=12)
        axes.set_xlabel("Product term name", fontsize=12)
        axes2.set_ylabel("Operating time (hours)", fontsize=12)
        axes2.set_ylim(ymax=425)
        axes.set_ylim(ymax=500)
        fig.suptitle("Average Fertiliser Input and Machinery & Irrigation "
                     f"Usage per ha, {source}", fontsize=14)
        fig.tight_layout()
        fig.savefig(f"{plot_save_directory}{source}"
                    "_fertiliser_operation_barchart",
                    bbox_inches='tight')
        

def map_sites(sources):

    '''
    Returns a plot of the latitude and longitude of all sites, 
    which is calculated from the region added.
    sources: list of source folders

    '''
    
    # Save directory for plots
    plot_save_directory = "plots/site/"
    os.makedirs(plot_save_directory, exist_ok=True)

    # Create figure
    fig = plt.figure(figsize=(7, 6))
    axes = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    long_list = []
    lat_list = []
    
    for source in sources:

        # Retrive paths, products and countries
        path_list, products, countries = get_sub_directories(
            f"sources/{source}/")
        
        for i, path in enumerate(path_list):
      
            # Retrieve arrays
            site_save_directory = f"arrays/{path}/site/lat_long.npy"

            # If files don't exist run processsing code
            if not os.path.exists(site_save_directory):
                request.get_lat_long_site(products[i], countries[i], source)
            
            site_array = np.load(site_save_directory, allow_pickle=True)
            site = pd.DataFrame.from_records(site_array)

            long_list.append(site['term.longitude'])
            lat_list.append(site["term.latitude"])

    # Flatten lists  
    long = list(itertools.chain.from_iterable(long_list))
    lat = list(itertools.chain.from_iterable(lat_list))
    
    # Put in dataframe, count repeats and add as new column
    long_lat = pd.DataFrame({'longitude': long, 'latitude': lat})
    long_lat = long_lat.groupby(long_lat.columns.tolist(),
                                as_index=False).size()

    # Plot background map
    fname = "reference/viz.GMRT_hillshade-color_thumbnail.png"
    img_extent = (68, 97, 7.5, 33)
    img = plt.imread(fname)
    gl = axes.gridlines(draw_labels=True, linewidth=2, color='white', # noqa F841
                        alpha=0, linestyle='--')  
    
    axes.imshow(img, origin='upper', extent=img_extent, 
                transform=ccrs.PlateCarree(), alpha=0.75)
    axes.coastlines(resolution='50m', color='black', linewidth=1)
    axes.add_feature(cf.BORDERS, color='black')
        
    # Plot points scaling by the frequency

    scatter = axes.scatter(long_lat['longitude'], long_lat['latitude'],
                           s=long_lat['size'], alpha=0.6, 
                           color="red", edgecolors="black")

    # Add scale legend
    plt.legend(*scatter.legend_elements("sizes", num=2, color='red'),
               loc="lower right", title="Scale", alignment='right', 
               handlelength=5, borderpad=2)

    # Figure alterations
    fig.suptitle("Regions Sampled, Scaled by Sample Number",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{plot_save_directory}_site_map",
                bbox_inches='tight')
    

def indiactor_barchart(sources, indicator, indicator_label):
    
    '''
    Returns a barchart of contributions to an indicator for each product.
    sources: list of source folders
    indicator: indicator
    indicator_label: Indicator label you want to appear on the plot
    (not necessarily the term id)
    country_split: optional argument, True if you want products split
    by country or False for them aggregated together, defaults to True
    '''
    
    # Save directory for plots
    plot_save_directory = "plots/impacts/"
    os.makedirs(plot_save_directory, exist_ok=True)

    # Load FAO csv for the labels
    fao_yield_df = pd.read_csv("reference/FAOSTAT_yields.csv")
    
    for source in sources:

        # Retrive paths, products and countries
        path_list, products, countries = get_sub_directories(
            f"sources/{source}/")
        
        # Create empty dataframes
        average_df = pd.DataFrame()
        average_inputs_df = pd.DataFrame()

        for i, path in enumerate(path_list):

            # Retrieve arrays
            impacts_save_directory = f"arrays/{path}/impacts/{
                indicator}/"

            # If files don't exist run processsing code
            if not os.path.exists(impacts_save_directory):
                request.get_indicator_contributions(
                    products[i], countries[i], source, indicator)

            group_emissions_array = np.load(
                f"{impacts_save_directory}group_emissions.npy")
            group_emissions = pd.DataFrame.from_records(
                group_emissions_array)
            
            group_input_emissions_array = np.load(
                f"{impacts_save_directory}group_input_emissions.npy")
            group_input_emissions = pd.DataFrame.from_records(
                group_input_emissions_array)
            
            # Retrieve the label from the FAO file
            fao_label = fao_yield_df[(fao_yield_df[
                'product'] == products[i]) & (fao_yield_df[
                    'country'] == countries[i])]["label"].iloc[0]
            
            # Append average of each emission to dataframe
            average_emissions = group_emissions.mean()
            average_df[fao_label] = average_emissions

            average_input_emissions = group_input_emissions.mean()
            average_inputs_df[fao_label] = average_input_emissions

        # Remove values below and above a resolution, and transpose

        resolution = (average_df.max().max())/50
        resolution = round(resolution, -int(floor(log10(abs(resolution)))))

        resolution_inputs = (average_inputs_df.max().max())/50
        resolution_inputs = round(resolution_inputs, -int(floor(log10(abs(
            resolution_inputs)))))
        
        average_df = average_df.loc[(average_df > resolution).any(
            axis=1) | (average_df < -resolution).any(axis=1)]
        average_df = average_df.transpose()
        
        average_inputs_df = average_inputs_df.loc[(
            average_inputs_df > resolution_inputs).any(axis=1) | (
                average_inputs_df < -resolution_inputs).any(axis=1)]
        average_inputs_df = average_inputs_df.transpose()
        
        # Plot values on a stacked bar chart
        colour_list = ['black', 'silver', 'lightcoral', 'darkred', 
                       'peachpuff', 'darkorange', 'gold', 'yellowgreen',
                       'green', 'turquoise', 'teal', 'deepskyblue', 
                       'royalblue', 'darkblue', 'mediumpurple', 'violet', 
                       'deeppink']
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 13))

        average_df.plot(kind='bar', stacked=True, ax=axes[0],
                        color=colour_list)
        average_inputs_df.plot(kind='bar', stacked=True, ax=axes[1],
                               color=colour_list)
    
        # Figure aleterations
        axes[0].set_ylabel(indicator_label, fontsize=14)
        axes[1].set_ylabel(indicator_label, fontsize=14)
        axes[0].set_xlabel("Product term name", fontsize=14)
        axes[1].set_xlabel("Product term name", fontsize=14)
        fig.suptitle(
            f"Indicator Contributions greater than {resolution}", 
            fontsize=12)
        fig.tight_layout()

        fig.savefig(
            f"{plot_save_directory}{source}_{indicator}_barchart.png", 
            bbox_inches='tight')


def normalised_inputs_boxplot(sources, upper_per=100, lower_per=0):

    '''
    Returns a boxplot for each input, with the x axis being the primary 
    products.
    self: list of product term.ids
    countries: list of country names
    sources: list of sources
    country_split: optional argument, True if you want products split
    by country or False for them aggregated together, defaults to True
    percentile: data outside this percentile is ignored
    '''

    # Save directory for plots
    plot_save_directory = "plots/cycle/inputs/"
    os.makedirs(plot_save_directory, exist_ok=True)

    for source in sources:
        countries = os.listdir(f"sources/{source}/")
        inputs_list = []

        for country in countries:
            products = os.listdir(f"sources/{source}/{country}")

            for index, product in enumerate(products):

                # Retrieve arrays
                array_save_directory = f"arrays/{source}/{country}/{
                    product}/cycle/input/normalised_all_inputs.npy"
            
                # If files don't exist run processsing code
                if not os.path.exists(array_save_directory):
                    request.get_inputs(product, country, source)
                    
                inputs_array = np.load(array_save_directory)
                inputs_list.append(inputs_array.dtype.names)
        
        # Find unique terms
        inputs_list = list(set(chain(*inputs_list)))
        
        # Create that number of subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(inputs_list), figsize=(
            7*len(inputs_list), 9))
        
        for c, country in enumerate(countries):
            products = os.listdir(f"sources/{source}/{country}")
            product_number = len(products)

            for p, product in enumerate(products):

                # Retrieve arrays
                array_save_directory = f"arrays/{source}/{country}/{
                    product}/cycle/input/normalised_all_inputs.npy"
            
                inputs_array = np.load(array_save_directory)
                inputs = pd.DataFrame.from_records(inputs_array)

                # Set values above the percentile to np.nan
                for j, column in enumerate(inputs):
                    upper = np.percentile(inputs[column], upper_per)
                    inputs.loc[inputs[column] > upper, column] = np.nan

                    lower = np.percentile(inputs[column], lower_per)
                    inputs.loc[inputs[column] < lower, column] = np.nan
                    
                # Plot for each input

                # Get position for plot
                position = c*product_number + p

                for i, input in enumerate(inputs_list):

                    # Check if the input is in the dataframe
                    columns_list = inputs.columns.values.tolist()

                    if input in columns_list:

                        # Figure alterations
                        axes[i].set_title(input, fontsize=12)
                        axes[i].set_xlabel("Products", fontsize=12)
                        axes[i].tick_params(axis='x', labelrotation=90)
                        axes[i].set_xlim(-0.5, 23.5)
                        axes[i].grid(True)

                        # Boxplot input, removing NaN values first
                        filtered_inputs = inputs[input][~np.isnan(inputs[input])]

                        axes[i].boxplot(filtered_inputs, showfliers=True, 
                                        positions=[position], widths=[0.75])
                       
                # Add blanks where the input isn't in the dataframe
                else:
                    axes[i].boxplot([0, 0], positions=[position], widths=[0.75])
        
                axes[i].set_xticks(range(0, product_number))
                # axes[i].set_xticklabels(self, fontsize=12)

        # Figure alterations
        fig.suptitle("Value inputted, normalised to the mean", fontsize=14)
        fig.tight_layout()
        fig.savefig(
            f"{plot_save_directory}{source}_{upper_per}_all_normalised", 
            bbox_inches='tight')


def variables_scatter_plot(sources, x_variable, y_variable, product):
    
    '''
    Returns a scatter plot of the variables against the impact chosen
    self: list of product term.ids
    sources: list of source folders
    indicator: indicators/endpoint
    variable: variable, i.e. name of a column in the variables dataframe

    '''
    
    # Save directory for plots
    plot_save_directory = "plots/impacts/"
    os.makedirs(plot_save_directory, exist_ok=True)

    # Create blank plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))

    for source in sources:
        countries = os.listdir(f"sources/{source}")

        for country in countries:

            # Retrieve arrays
            variables_save_directory = f"arrays/{source}/{
                country}/{product}/cycle/variables.csv"
            
            # If files don't exist run processsing code
            if not os.path.exists(variables_save_directory):
                request.get_variables_impacts(
                    product, country, source)

            # Retrieve arrays
            variables = pd.read_csv(variables_save_directory)
            
            # Scatter plot the x and y variable
            x_points = variables[x_variable]
            y_points = variables[y_variable]

            axes.scatter(x_points, y_points)
    
    # Add labels to plot and save
    axes.set_xlabel(x_variable, fontsize=12)
    axes.set_ylabel(y_variable, fontsize=12)

    fig.tight_layout()
    fig.savefig(
            f"{plot_save_directory}_{product}_{x_variable}_{y_variable}.png", 
            bbox_inches='tight')
                
                
"""
@author: lgonzalez
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import arviz as az
import pymc as pm
from netCDF4 import Dataset as NetCDFFile

from biosc.preprocessing import Preprocessing
from biosc.bhm import BayesianModel


class BayesianModelPlots:
    def __init__(self, data_file, priors, file, path_data, path_models, L, ages, colormap):
        self.data_file = data_file
        self.priors = priors
        self.file = file
        self.path_data = path_data
        self.path_models = path_models
        self.L = L
        self.ages = ages
        self.colormap = colormap

    def process_idata(self, plot_type='all', band=None, save=None, ifile=None, pfile=None):
        directory = os.getcwd()
        folder = 'idata'
        self.path_folder = os.path.join(directory, folder)
        self.path_file = os.path.join(self.path_folder, self.file)

        directory = os.path.normpath(directory)
        self.path_folder = os.path.normpath(self.path_folder)
        self.path_file = os.path.normpath(self.path_file)

        print(self.path_file)
        
        data_test = NetCDFFile(self.path_file, 'r')

        posterior_group = data_test.groups['posterior']

        Li_data = posterior_group.variables['Li*'][:]

        Li_samples = posterior_group.variables['Li*'][:]

        Li_samples_reshaped = Li_samples.reshape(-1, self.L)

        i_mean = np.mean(Li_samples_reshaped, axis=0)

        Li_mean_simple = np.mean(Li_samples_reshaped, axis=0).data

        Li_summary = pm.summary(posterior_group.variables['Li*'][:])

        M_data = posterior_group.variables['M*'][:]

        G_data = M_data[..., 0]
        BP_data = M_data[..., 1]
        RP_data = M_data[..., 2]
        J_data = M_data[..., 3]
        H_data = M_data[..., 4]
        K_data = M_data[..., 5]
        g_data = M_data[..., 6]
        r_data = M_data[..., 7]
        i_data = M_data[..., 8]
        y_data = M_data[..., 9]
        z_data = M_data[..., 10]

        J_samples = J_data.reshape(-1, self.L)
        J_samples_mean = np.mean(J_samples, axis=0)
        K_samples = K_data.reshape(-1, self.L)
        K_samples_mean = np.mean(K_samples, axis=0)
        G_samples = G_data.reshape(-1, self.L)
        G_samples_mean = np.mean(G_samples, axis=0)
        BP_samples = BP_data.reshape(-1, self.L)
        BP_samples_mean = np.mean(BP_samples, axis=0)
        RP_samples = RP_data.reshape(-1, self.L)
        RP_samples_mean = np.mean(RP_samples, axis=0)
        H_samples = H_data.reshape(-1, self.L)
        H_samples_mean = np.mean(H_samples, axis=0)
        g_samples = g_data.reshape(-1, self.L)
        g_samples_mean = np.mean(g_samples, axis=0)
        r_samples = r_data.reshape(-1, self.L)
        r_samples_mean = np.mean(r_samples, axis=0)
        i_samples = i_data.reshape(-1, self.L)
        i_samples_mean = np.mean(i_samples, axis=0)
        y_samples = y_data.reshape(-1, self.L)
        y_samples_mean = np.mean(y_samples, axis=0)
        z_samples = z_data.reshape(-1, self.L)
        z_samples_mean = np.mean(z_samples, axis=0)
        
        # prepare photometric bands
        mags = ['G', 'BP', 'RP', 'J', 'H', 'K', 'g', 'r', 'i', 'y', 'z']
        bands = ['BP', 'RP', 'J', 'H', 'K', 'r', 'i', 'y', 'z']
        bands_e = ['bp', 'rp', 'Jmag', 'Hmag', 'Kmag', 'rmag', 'imag', 'ymag', 'zmag']

        
        # load Pleiades and model
        Pleiades_Li = self.load_pleiades_data(self.path_data)
        model_Li_isochrones = self.load_model_isochrones(self.path_models)
        
        
        plt.rcParams.update({'font.size': 14})
        
        # bayesian inference for data
        data_model, prior_sample = self.bayesian_model(save, ifile, pfile)
        
        # load prior sample data
        dataframe_prior_sample = prior_sample.prior['M*'].to_dataframe()
        
        data_model_pd = pd.DataFrame(data_model['m_data']['data'])
        
        data_model_pd.rename(columns={'g': 'G', 'bp': 'BP', 'rp': 'RP', 'Jmag': 'J', 'Hmag': 'H', 'Kmag': 'K',
                                    'gmag': 'g', 'rmag': 'r', 'imag': 'i', 'ymag': 'y', 'zmag': 'z'}, inplace=True)
        
        data_model_pd['Parallax'] = data_model['parallax_data']['data']
        data_model_pd['Li_data'] = data_model['Li_data']['data']
        
        items = data_model.items()
        for key, value in items:
            print(f'Key {key}')
        
        # get abs magnitudes and distance corrections
        Pleiades_Li, data_model_pd = self.calculate_absolute_magnitudes(Pleiades_Li, data_model_pd, mags, bands)
        
        # Li plot
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        self.plot_additional_subplot(ax2, Pleiades_Li, data_model_pd, model_Li_isochrones, G_samples_mean, J_samples_mean, Li_mean_simple, self.ages, self.colormap)
        ax2.legend(loc='lower right')
        plt.show()
        
        # plot subplots or single band plot
        if plot_type == 'all':
            plt.rcParams.update({'font.size': 24})  # update font size for subplots
            fig, axs = plt.subplots(3, 3, figsize=(24, 18))
            self.plot_subplots(axs, bands, bands_e, Pleiades_Li, data_model_pd, model_Li_isochrones, self.ages, self.colormap,
                            G_samples_mean, BP_samples_mean, RP_samples_mean, J_samples_mean, H_samples_mean,
                            K_samples_mean, g_samples_mean, r_samples_mean, i_samples_mean, y_samples_mean, z_samples_mean)
            plt.tight_layout()
            plt.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.5, 1.5))
            plt.show()
        elif plot_type == 'unique':
            plt.rcParams.update({'font.size': 14})  # update font size for individual plot
            if band is None or band not in mags:
                print("Please provide a valid photometric band.")
                return None
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                self.plot_single_band(ax, Pleiades_Li, data_model_pd, model_Li_isochrones, self.ages, self.colormap, band)
                plt.legend(loc='upper right', fontsize=12)
                plt.show()
        else:
            plt.rcParams.update({'font.size': 14})  # update font size for Li plot
            print("Invalid plot type. Please choose 'all' or 'unique'.")
            return None

        return axs, ax2

    def bayesian_model(self, save=False, ifile=None, pfile=None):
        prep = Preprocessing(self.data_file, sortPho=False)
        parallax_data = prep.get_parallax()
        Li_data = prep.get_Li()
        m_data = prep.get_magnitude(fillna='max')

        model = BayesianModel(parallax_data, m_data, Li_data)
        model.compile(self.priors, POPho=False, POLi=False)
        model.sample(draws=2000, chains=4)
        model.sample_posterior_predictive()

        prior_sample = model.sample_prior_predictive(samples=500, return_inferencedata=True)

        if save:
            model.save(ifile)
            prior_sample.to_netcdf(pfile)

        model.plot_trace(var_names=['Age [Myr]', 'Distance [pc]'])

        data_model = model.generate_data(mode='dist')

        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        az.plot_ppc(model.idata, var_names='parallax [mas]', ax=axs[0])
        az.plot_ppc(model.idata, var_names=r'flux [erg s$^{-1}$ cm$^{-2}$]', ax=axs[1])
        az.plot_ppc(model.idata, var_names='A(Li) [dex]', ax=axs[2])
        plt.show()

        # Plot QQ for parallax
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_ylabel('Observed parallax [mas]')
        ax.set_xlabel('Parallax [mas]')
        model.plot_QQ('parallax [mas]', fig, ax)
        plt.show()

        # Plot QQ for Li
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_ylabel('Observed A(Li) [dex]')
        ax.set_xlabel('A(Li) [dex]')
        model.plot_QQ('A(Li) [dex]', fig, ax)
        plt.show()

        az.plot_posterior(model.trace, var_names=['Age [Myr]'])

        return data_model, prior_sample

    def load_pleiades_data(self, path_data):
        '''
        Loads Pleiades data from a CSV file.
        '''
        Pleiades_Li = pd.read_csv(path_data)
        Pleiades_Li['e_bp'] = Pleiades_Li['bp_error']
        Pleiades_Li['e_rp'] = Pleiades_Li['rp_error']
        
        #rename bands
        Pleiades_Li.rename(columns={'g': 'G', 'bp': 'BP', 'rp': 'RP', 'Jmag': 'J', 'Hmag': 'H', 'Kmag': 'K',
                                    'gmag': 'g', 'rmag': 'r', 'imag': 'i', 'ymag': 'y', 'zmag': 'z'}, inplace=True)
        
        return Pleiades_Li

    def load_model_isochrones(self, path_models):
        '''
        Loads model isochrones data from a CSV file.
        '''
        model = pd.read_csv(path_models)
        
        model.rename(columns={'age_Myr': 'age_Gyr',
                            'Teff(K)': 'Teff',
                            'G_RP': 'RP',
                            'G_BP': 'BP',
                            'r_p1': 'r',
                            'i_p1': 'i',
                            'y_p1': 'y',
                            'z_p1': 'z'}, inplace=True)
        
        model['age_Gyr'] *= 0.001
        model['A(Li)'] = np.log10(model['Li']) + 3.3
        
        # Create a dictionary to store dataframes for each isochrone
        model_Li_isochrones = {}

        # Loop over each row in the model dataframe
        for index, row in model.iterrows():
            # Get the value of age_Gyr from the current row
            age_Gyr = row['age_Gyr']

            # Check if the value of age_Gyr already exists as a key in the dictionary
            if age_Gyr not in model_Li_isochrones:
                # If it doesn't exist, create a new entry in the dictionary with the value of age_Gyr as the key
                model_Li_isochrones[age_Gyr] = []

            # Add the current row to the corresponding value of age_Gyr in the dictionary
            model_Li_isochrones[age_Gyr].append(row)

        # Convert each list of rows into a dataframe and replace the list in the dictionary
        for age_Gyr, rows in model_Li_isochrones.items():
            model_Li_isochrones[age_Gyr] = pd.DataFrame(rows)
        
        return model_Li_isochrones

    def calculate_absolute_magnitudes(self, Pleiades_Li, data_model_pd, mags, bands):
        '''
        Calculates absolute magnitudes and distance corrections.
        '''
        distance = 1 / (data_model_pd['Parallax'] * 1e-3)
        distance_mod = 5 * np.log10(distance) - 5 
        Mags = []
        for i, mag in enumerate(mags):
            Pleiades_Li[mag+'_abs'] = Pleiades_Li[mag] - distance_mod
            data_model_pd[mag+'_abs'] = data_model_pd[mag] - distance_mod
            Mags.append(mag+'_abs')
        return Pleiades_Li, data_model_pd

    def plot_subplots(self, axs, bands, bands_e, Pleiades_Li, data_model_pd, model_Li_isochrones, ages, colormap, 
                    G_samples_mean, BP_samples_mean, RP_samples_mean, J_samples_mean, H_samples_mean, 
                    K_samples_mean, g_samples_mean, r_samples_mean, i_samples_mean, y_samples_mean, z_samples_mean):
        '''
        Plots subplots for each photometric band.
        '''
        
        colors = self.get_colors(ages, colormap)
        for i, band in enumerate(bands):
            diff_obs = Pleiades_Li['G_abs'] - Pleiades_Li[band+'_abs']
            diff_model = data_model_pd['G_abs'] - data_model_pd[band+'_abs']
            ax = axs[i // 3, i % 3]
            self.plot_observation(ax, diff_obs, Pleiades_Li, band, bands_e[i])
            self.plot_model(ax, diff_model, data_model_pd, band, model_Li_isochrones, ages, colors,
                        G_samples_mean, BP_samples_mean, RP_samples_mean, J_samples_mean, H_samples_mean, 
                        K_samples_mean, g_samples_mean, r_samples_mean, i_samples_mean, y_samples_mean, z_samples_mean)
            ax.set_xlabel(f'G-{band} [mag]')
            ax.set_ylabel(f'{band} [mag]')
            ax.invert_yaxis()
            if band in ['BP', 'r']:
                ax.invert_xaxis()

    def plot_model(self, ax, diff_model, data_model_pd, band, model_Li_isochrones, ages, colors,
                G_samples_mean, BP_samples_mean, RP_samples_mean, J_samples_mean, H_samples_mean, 
                K_samples_mean, g_samples_mean, r_samples_mean, i_samples_mean, y_samples_mean, z_samples_mean):
        ax.scatter(diff_model, data_model_pd[band+'_abs'], s=5, label='Prior')
        sample_mean_band = locals()[f"{band}_samples_mean"]
        ax.scatter(G_samples_mean - sample_mean_band, sample_mean_band, s=5, label='Post', color='red')
        for age, color in zip(ages, colors):
            ax.plot(model_Li_isochrones[age]['G'] - model_Li_isochrones[age][band], 
                    model_Li_isochrones[age][band], linewidth=1, label=f'{age} Gyr', color=color)

    def get_colors(self, ages, colormap):
        '''
        Generates colors based on colormap and ages.
        '''
        min_age = min(ages)
        max_age = max(ages)
        num_ages = len(ages)
        
        norm = plt.Normalize(min_age, max_age)
        colormap_function = cm.get_cmap(colormap)
        
        #linespaced points
        age_points = np.linspace(min_age, max_age, num_ages)
        
        
        return [colormap_function(norm(age)) for age in age_points]

    def plot_observation(self, ax, diff_obs, Pleiades_Li, band, band_error):
        ax.errorbar(diff_obs, Pleiades_Li[band+'_abs'], yerr=Pleiades_Li['e_'+band_error], fmt='.', capsize=0, linewidth=1, capthick=1, label='Obs', color='orange', zorder=1)


    def plot_additional_subplot(self, ax2, Pleiades_Li, data_model_pd, model_Li_isochrones, G_samples_mean, J_samples_mean, Li_mean_simple, ages, colormap):
        colors = self.get_colors(ages, colormap)
        ax2.errorbar(Pleiades_Li['G'] - Pleiades_Li['J'], Pleiades_Li['ALi'], yerr=Pleiades_Li['e_ALi'], fmt='.', capsize=0, linewidth=1, capthick=1, label='Obs', color='orange', zorder=1)
        ax2.scatter(data_model_pd['G'] - data_model_pd['J'], data_model_pd['Li_data'], s=5, label='Prior', zorder=2)
        for age, color in zip(ages, colors):
            ax2.plot(model_Li_isochrones[age]['G'] - model_Li_isochrones[age]['J'], 
                    model_Li_isochrones[age]['A(Li)'], linewidth=1, label=f'{age} Gyr', color=color)
        ax2.set_xlabel('G-J [mag]')
        ax2.set_ylabel('A(Li) [dex]')
        ax2.scatter(G_samples_mean - J_samples_mean, Li_mean_simple, s=5, label='Post', color='red', zorder=3)
        

    def plot_single_band(self, ax, Pleiades_Li, data_model_pd, model_Li_isochrones, ages, colormap, band, bands, band_e):
        colors = self.get_colors(ages, colormap)
        diff_obs = Pleiades_Li['G'] - Pleiades_Li[band]
        diff_model = data_model_pd['G_abs'] - data_model_pd[band+'_abs']
        self.plot_observation(ax, diff_obs, Pleiades_Li, band, band_e[bands.index(band)])
        self.plot_model(ax, diff_model, data_model_pd, band, model_Li_isochrones, ages, colors)
        ax.set_xlabel(f'G-{band} [mag]')
        ax.set_ylabel(f'{band} [mag]')
        ax.invert_yaxis()
        if band in ['BP', 'r']:
            ax.invert_xaxis()
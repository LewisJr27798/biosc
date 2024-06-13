import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import pymc as pm
from pymc import HalfCauchy, Model, Normal, sample
import arviz as az
import bambi as bmb
import xarray as xr
import os
import shutil
import matplotlib.cm as cm
from netCDF4 import Dataset as NetCDFF
from scipy.stats import gaussian_kde
import logging
import read_mist_models
import time



__all__ = ['plt', 'np', 'pd', 'pc']

def pc(string):
    """

    Args:
    string : for terminal used: work or home; string format.

    Returns:
    path_all : path to all files and environment.

    """
    if string == 'home':
        path_all = 'C:/Users/lgrjr/OneDrive/Escritorio/CAB_INTA-CSIC/01-Doctorado/013-Jupyter/0134-biosc_env/'
    elif string == 'work':
        path_all = '/pcdisk/dalen/lgonzalez/OneDrive/Escritorio/CAB_INTA-CSIC/01-Doctorado/013-Jupyter/0134-biosc_env/'
    return path_all

def pc_other(string_dir):
    path_all = string_dir
    return path_all

def select_nearest_age(dic, desired_age):
    """
    Args:
        desired_age (float)
    
    Returns:
        float: nearest_age
    """
    available_ages = list(dic.keys())
    closest_age = min(available_ages, key=lambda x: abs(x - desired_age))
    return closest_age

class PleiadesData:
    def __init__(self, path):
        self.path_data = path
        self.data = pd.read_csv(self.path_data)
        self._rename_columns()
        self._calculate_distances()
        self._calculate_absolute_magnitudes()
        self._calculate_colors()
        self._calculate_luminosity()

    def _rename_columns(self):
        self.data.rename(columns={'g_error': 'e_g', 'rp_error': 'e_rp', 'bp_error': 'e_bp'}, inplace=True)

    def _calculate_distances(self):
        self.data['distance'] = 1 / (self.data['parallax'] * 1e-3)
        self.data['distance_modulus'] = 5 * np.log10(self.data['distance']) - 5

    def _calculate_absolute_magnitudes(self):
        distance_mod_obs = self.data['distance_modulus']
        self.data['G_abs'] = self.data['g'] - distance_mod_obs
        self.data['BP_abs'] = self.data['bp'] - distance_mod_obs
        self.data['RP_abs'] = self.data['rp'] - distance_mod_obs
        self.data['J_abs'] = self.data['Jmag'] - distance_mod_obs
        self.data['H_abs'] = self.data['Hmag'] - distance_mod_obs
        self.data['K_abs'] = self.data['Kmag'] - distance_mod_obs
        self.data['r_abs'] = self.data['rmag'] - distance_mod_obs
        self.data['i_abs'] = self.data['imag'] - distance_mod_obs
        self.data['y_abs'] = self.data['ymag'] - distance_mod_obs
        self.data['z_abs'] = self.data['zmag'] - distance_mod_obs
        self.data['g_abs'] = self.data['gmag'] - distance_mod_obs

    def _calculate_colors(self):
        self.data['G-J'] = self.data['G_abs'] - self.data['J_abs']
        self.data['G-RP'] = self.data['G_abs'] - self.data['RP_abs']
        self.data['BP-RP'] = self.data['BP_abs'] - self.data['RP_abs']

    def _calculate_luminosity(self):
        Tsun = 5772
        self.data['Lsun'] = self.data['Rad']**2 * (self.data['Teff_x'] / Tsun)**4
        self.data['log(L/Lsun)'] = np.log10(self.data['Lsun'])
        

class Models:
    class PARSEC:
        def __init__(self, path_all, libs=['Gaia', '2MASS', 'PanSTARRS']):
            self.path_all = path_all
            self.libs = libs
            self.PARSEC_iso_omega_00_Phot = {}
            self.PARSEC_iso_omega_00_sun_Phot = {}
            self.PARSEC_iso_omega_00_Phot_dict = {}
            self.PARSEC_iso_omega_00_sun_Phot_dict = {}
            self.common_columns = None
            self._process_files()
            self._concatenate_data()
            self._generate_dicts()

        def _process_files(self):
            for l in self.libs:
                parsec_file = f'{self.path_all}data/PARSEC_iso_omega_00_{l}.dat'
                data = np.loadtxt(parsec_file)
                
                with open(parsec_file, 'r') as file:
                    for i, line in enumerate(file):
                        if i == 14:
                            columns = line.strip().lstrip('#').split()
                            break
                
                dataframe = pd.DataFrame(data, columns=columns)
                dataframe.rename(columns={'Mass': 'M/Ms'}, inplace=True)
                dataframe['Age [Gyr]'] = (10**dataframe['logAge']) / (1e9)
                
                self.PARSEC_iso_omega_00_Phot[l] = dataframe
                
                parsec_sun_file = f'{self.path_all}data/PARSEC_iso_omega_00_sun_{l}.dat'
                data_sun = np.loadtxt(parsec_sun_file)
                
                with open(parsec_sun_file, 'r') as file:
                    for i, line in enumerate(file):
                        if i == 14:
                            columns_sun = line.strip().lstrip('#').split()
                            break
                
                dataframe_sun = pd.DataFrame(data_sun, columns=columns_sun)
                dataframe_sun.rename(columns={'Mass': 'M/Ms'}, inplace=True)
                dataframe_sun['Age [Gyr]'] = (10**dataframe_sun['logAge']) / (1e9)
                
                self.PARSEC_iso_omega_00_sun_Phot[l] = dataframe_sun
            return self.PARSEC_iso_omega_00_Phot, self.PARSEC_iso_omega_00_sun_Phot

        def _concatenate_data(self):
            dataframes_concatenate = [df for df in self.PARSEC_iso_omega_00_Phot.values()]
            dataframe_final = pd.concat(dataframes_concatenate, axis=1)
            dataframe_final = dataframe_final.loc[:, ~dataframe_final.columns.duplicated()]

            csv_file_parsec = f'{self.path_all}data/PARSEC_iso_omega_00_Phot.csv'
            dataframe_final.to_csv(csv_file_parsec, index=False)
            self.dataframe_final = dataframe_final

            dataframes_concatenate_sun = [df for df in self.PARSEC_iso_omega_00_sun_Phot.values()]
            dataframe_final_sun = pd.concat(dataframes_concatenate_sun, axis=1)
            dataframe_final_sun = dataframe_final_sun.loc[:, ~dataframe_final_sun.columns.duplicated()]

            csv_file_parsec_sun = f'{self.path_all}data/PARSEC_iso_omega_00_sun_Phot.csv'
            dataframe_final_sun.to_csv(csv_file_parsec_sun, index=False)
            self.dataframe_final_sun = dataframe_final_sun

        def _generate_dicts(self):
            for age in self.dataframe_final['Age [Gyr]'].unique():
                dataframe_by_age = self.dataframe_final[self.dataframe_final['Age [Gyr]'] == age]
                dataframe_by_age = dataframe_by_age.loc[dataframe_by_age['G_i00'] > 0].reset_index(drop=True)
                dataframe_by_age['Teff'] = 10**dataframe_by_age['logTe']
                dataframe_by_age['Lsun'] = 10**dataframe_by_age['logL']
                self.PARSEC_iso_omega_00_Phot_dict[age] = dataframe_by_age

            for age in self.dataframe_final_sun['Age [Gyr]'].unique():
                dataframe_by_age = self.dataframe_final_sun[self.dataframe_final_sun['Age [Gyr]'] == age]
                dataframe_by_age = dataframe_by_age.loc[dataframe_by_age['G_i00'] > 0].reset_index(drop=True)
                dataframe_by_age['Teff'] = 10**dataframe_by_age['logTe']
                dataframe_by_age['Lsun'] = 10**dataframe_by_age['logL']
                self.PARSEC_iso_omega_00_sun_Phot_dict[round(age, 3)] = dataframe_by_age
            return self.PARSEC_iso_omega_00_Phot_dict, self.PARSEC_iso_omega_00_sun_Phot_dict


        def get_dataframe(self, sun=False):
            if sun:
                return self.dataframe_final_sun
            else:
                return self.dataframe_final

        def get_dataframe_by_age(self, age, sun=False):
            if sun:
                age_dict = self.PARSEC_iso_omega_00_sun_Phot_dict
            else:
                age_dict = self.PARSEC_iso_omega_00_Phot_dict
            
            closest_age = min(age_dict.keys(), key=lambda k: abs(k - age))
            return age_dict.get(closest_age)
                
    class BTSettl:
        def __init__(self, path):
            self.path = path
            self.BTSettl = None
            self.BTSettl_Li_isochrones = {}
            self._load_data()
            self._process_data()
            self._create_isochrones_dict()
    
        def _load_data(self):
            self.BTSettl = pd.read_csv(self.path)
    
        def _process_data(self):
            self.BTSettl.rename(columns={
                'age_Myr': 'age_Gyr',
                'Teff(K)': 'Teff',
                'L/Ls': 'log(L/Lsun)',
                'G': 'G_abs',
                'G_RP': 'RP_abs',
                'G_BP': 'BP_abs',
                'J': 'J_abs',
                'K': 'K_abs',
                'H': 'H_abs',
                'g_p1': 'g_abs',
                'r_p1': 'r_abs',
                'i_p1': 'i_abs',
                'y_p1': 'y_abs',
                'z_p1': 'z_abs'}, inplace=True)
    
            self.BTSettl['age_Gyr'] *= 0.001
            with np.errstate(divide='ignore', invalid='ignore'):
                self.BTSettl['A(Li)'] = np.log10(self.BTSettl['Li']) + 3.3
            self.BTSettl['Lsun'] = 10**self.BTSettl['log(L/Lsun)']
    
        def _create_isochrones_dict(self):
            for index, row in self.BTSettl.iterrows():
                age_Gyr = row['age_Gyr']
                if age_Gyr not in self.BTSettl_Li_isochrones:
                    self.BTSettl_Li_isochrones[age_Gyr] = []
                self.BTSettl_Li_isochrones[age_Gyr].append(row)
    
            for age_Gyr, rows in self.BTSettl_Li_isochrones.items():
                self.BTSettl_Li_isochrones[age_Gyr] = pd.DataFrame(rows)
            return self.BTSettl_Li_isochrones
    
        def get_dataframe(self):
            return self.BTSettl

        def get_dataframe_by_age(self, age):
            closest_age = min(self.BTSettl_Li_isochrones.keys(), key=lambda k: abs(k - age))
            return self.BTSettl_Li_isochrones.get(closest_age)
        
    class MIST:
        def __init__(self, path_all):
            self.path_all = path_all
        
        def isocmd_to_dataframe(self, isocmd_set, ages):
            data_dict = {}
            for age, isocmd in zip(ages, isocmd_set):
                data_dict[age] = pd.DataFrame(isocmd)
            return data_dict
    
        def iso_to_dataframe(self, iso_set, ages):
            data_dict = {}
            for age, iso in zip(ages, iso_set):
                data_dict[age] = pd.DataFrame(iso)
            return data_dict
    
        def files_show(self, path):
            if os.path.isdir(path):
                files = os.listdir(path)
            else:
                print('Error')
    
        def PG2_files(self, vvcrit):
            vvcrit = str(vvcrit)
            path_G2 = self.path_all + 'data/MIST_v1.2_vvcrit'+vvcrit+'_UBVRIplus/'
            self.files_show(path_G2)
            path_P = self.path_all + 'data/MIST_v1.2_vvcrit'+vvcrit+'_PanSTARRS/'
            self.files_show(path_P)
            path = self.path_all + 'data/MIST_v1.2_vvcrit'+vvcrit+'_full_isos/'
            self.files_show(path)
            return path_G2, path_P, path
    
        def file_copy(self, phot, vvcrit, feh):
            vvcrit = str(vvcrit)
            source_directory = self.path_all + 'data/MIST_v1.2_vvcrit'+vvcrit+'_'+phot+'/'
            target_directory = self.path_all + 'data/'
            file_name = 'MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit0.0_'+phot+'.iso.cmd'
            source_path = os.path.join(source_directory, file_name)
            file_name = os.path.basename(source_path)
            target_path = os.path.join(target_directory, file_name)
            shutil.copy(source_path, target_path)
    
        def read_iso(self, phot_G2, phot_P, vvcrit, feh):
            vvcrit = str(vvcrit)
            filename_iso_G2 = os.path.join(self.path_all, 'data', 'MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_'+phot_G2+'.iso.cmd')
            filename_iso_P = os.path.join(self.path_all, 'data', 'MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_'+phot_P+'.iso.cmd')
            filename_iso_full = os.path.join(self.path_all, 'data', 'MIST_v1.2_vvcrit0.0_full_isos', 'MIST_v1.2_feh_'+feh+'_afe_p0.0_vvcrit'+vvcrit+'_full.iso')
            iso_P = read_mist_models.ISOCMD(filename_iso_P)
            iso_G2 = read_mist_models.ISOCMD(filename_iso_G2)
            iso = read_mist_models.ISO(filename_iso_full)
            ages_P_sorted = sorted(iso_P.ages)
            ages_G2_sorted = sorted(iso_G2.ages)
            ages_sorted = sorted(iso.ages)
            MIST_P = self.isocmd_to_dataframe(iso_P.isocmds, ages_P_sorted)
            MIST_G2 = self.isocmd_to_dataframe(iso_G2.isocmds, ages_G2_sorted)
            MIST = self.iso_to_dataframe(iso.isos, ages_sorted)
            MIST_P = {round((10 ** age) / 1e9, 4): df for age, df in MIST_P.items()}
            MIST_G2 = {round((10 ** age) / 1e9, 4): df for age, df in MIST_G2.items()}
            MIST = {round((10 ** age) / 1e9, 4): df for age, df in MIST.items()}
            MIST_FULL = {}
            for age in MIST.keys():
                if age in MIST_G2 and MIST_P:
                    df_P = MIST_P[age]
                    df_G2 = MIST_G2[age]
                    df = MIST[age]
                    combined_df = pd.concat([df_P.iloc[:, :9], df_P.iloc[:, 9:], df_G2.iloc[:, 9:], df['surface_li7']], axis=1)
                    combined_df = combined_df.loc[combined_df['Gaia_G_EDR3'] > 0].reset_index(drop=True)
                    combined_df['Teff'] = 10**combined_df['log_Teff']
                    combined_df['Lsun'] = 10**combined_df['log_L']
                    MIST_FULL[age] = combined_df
    
            for age, df in MIST_FULL.items():
                column_mapping = {'star_mass': 'M/Ms',
                                'Gaia_G_EDR3': 'G',
                                'Gaia_BP_EDR3': 'BP',
                                'Gaia_RP_EDR3': 'RP',
                                '2MASS_J': 'J',
                                '2MASS_H': 'H',
                                '2MASS_Ks': 'K',
                                'PS_g': 'g',
                                'PS_r': 'r',
                                'PS_i': 'i',
                                'PS_z': 'z',
                                'PS_y': 'y',
                                'PS_w': 'w'
                            }
                MIST_FULL[age] = df.rename(columns=column_mapping)
    
            print('version: ', iso.version)
            print('abundances: ', iso.abun)
            print('rotation: ', iso.rot)
    
            return MIST_FULL
        
    class SPOTS:
        def __init__(self, path_all):
            self.directory = os.path.join(path_all, 'data/SPOTS_iso/')
            self.SPOTS = self.read_SPOTS_files()
    
        def read_SPOTS_files(self):
            SPOTS_data = {}
            for filename in os.listdir(self.directory):
                if filename.endswith(".isoc"):
                    file_path = os.path.join(self.directory, filename)
                    key = filename.split(".")[0]
                    age_dataframes = self.extract_dataframes(file_path)
                    SPOTS_data[key] = age_dataframes
            return SPOTS_data
    
        def extract_dataframes(self, file_path):
            age_dataframes = {}
            with open(file_path, 'r') as file:
                data_lines = []
                for line in file:
                    if not line.startswith("##") and line.strip():
                        data = line.strip().split()
                        if len(data) > 1:
                            data = [float(value) if value != '-99.0000' else None for value in data]
                            data_lines.append(data)
                df = pd.DataFrame(data_lines, columns=["log10_Age(yr)", "Mass", "Fspot", "Xspot", "log(L/Lsun)", "log(R/Rsun)", "log(g)", "log(Teff)", "log(T_hot)", "log(T_cool)", "TauCZ", "Li/Li0", "B_mag", "V_mag", "Rc_mag", "Ic_mag", "J_mag", "H_mag", "K_mag", "W1_mag", "G_mag", "BP_mag", "RP_mag"])
                df['Age [Gyr]'] = round((10**(df['log10_Age(yr)'])) / (1e9), 3)
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['A(Li)'] = np.log10(df['Li/Li0']) + 3.3
                df['M/Ms'] = df['Mass']
                df['Teff'] = 10**df['log(Teff)']
                df['Lsun'] = 10**df['log(L/Lsun)']
                df['G'] = df['G_mag']
                df['BP'] = df['BP_mag']
                df['RP'] = df['RP_mag']
                df.drop(columns=['log10_Age(yr)'], inplace=True)
                grouped = df.groupby('Age [Gyr]')
                for age, group in grouped:
                    age_dataframes[age] = group.reset_index(drop=True)
            return age_dataframes
        
    class SPOTS_YBC:
        def __init__(self, path_all):
            self.path_all = path_all
            self.spots_f000_edr3 = None
            self.SPOTS_edr3 = None
            self._process_data()
    
        def _ave_flux(self, mcool, mhot, Thot, Tcool, fspot):
            fs = fspot
            fp = fs / (fs + (1-fs)*(Tcool/Thot)**4)
            M_ave = -2.5*np.log10((1-fp)*10**(-mhot/2.5) + fp*10**(-mcool/2.5))
            return M_ave
    
        def _process_data(self):
            f_array = ['00', '17', '34', '51', '68', '85']
            fspots = [0.00, 0.17, 0.34, 0.51, 0.68, 0.85]
            self.spots_f000_edr3 = pd.read_csv(self.path_all + 'data/f000_ybc_2m_edr3.txt', sep='\s+')
            self.SPOTS_edr3 = {}
            for f, f_i in zip(f_array, fspots):
                self.SPOTS_edr3[f] = {}
                path = self.path_all + 'data/f0' + f + '_ybc_2m_edr3.txt'
                dataframe = pd.read_csv(path, sep='\s+')
                dataframe['Age_Gyr'] = 10**dataframe['logAge']/1e9
                dataframe['BP_abs'] = self._ave_flux(dataframe['G_BP_cool'], dataframe['G_BP_hot'], 10**dataframe['log(T_hot)'], 10**dataframe['log(T_cool)'], f_i)
                dataframe['RP_abs'] = self._ave_flux(dataframe['G_RP_cool'], dataframe['G_RP_hot'], 10**dataframe['log(T_hot)'], 10**dataframe['log(T_cool)'], f_i)
                dataframe['G_abs'] = self._ave_flux(dataframe['G_cool'], dataframe['G_hot'], 10**dataframe['log(T_hot)'], 10**dataframe['log(T_cool)'], f_i)
                dataframe['J_abs'] = self._ave_flux(dataframe['J_cool'], dataframe['J_hot'], 10**dataframe['log(T_hot)'], 10**dataframe['log(T_cool)'], f_i)
                dataframe['H_abs'] = self._ave_flux(dataframe['H_cool'], dataframe['H_hot'], 10**dataframe['log(T_hot)'], 10**dataframe['log(T_cool)'], f_i)
                dataframe['K_abs'] = self._ave_flux(dataframe['Ks_cool'], dataframe['Ks_hot'], 10**dataframe['log(T_hot)'], 10**dataframe['log(T_cool)'], f_i)
                dataframe['M/Ms'] = dataframe['Mass']
                dataframe['BP-RP'] = dataframe['BP_abs'] - dataframe['RP_abs']
                dataframe['G-J'] = dataframe['G_abs'] - dataframe['J_abs']
                dataframe['G-RP'] = dataframe['G_abs'] - dataframe['RP_abs']
                dataframe['J-H'] = dataframe['J_abs'] - dataframe['H_abs']
                dataframe['H-K'] = dataframe['H_abs'] - dataframe['K_abs']
                dataframe['G-H'] = dataframe['H_abs'] - dataframe['H_abs']
                dataframe['G-K'] = dataframe['G_abs'] - dataframe['K_abs']
                dataframe['G-V'] = dataframe['G_abs'] - dataframe['V_mag']
                with np.errstate(divide='ignore', invalid='ignore'):
                    dataframe['A(Li)'] = np.log10(dataframe['Li/Li0']) + 3.3
                dataframe['Lsun'] = 10**dataframe['log(L/Lsun)']
                dataframe['Teff'] = 10**dataframe['log(Teff)']
                for age_gyr, df_group in dataframe.groupby('Age_Gyr'):
                    self.SPOTS_edr3[f][round(age_gyr,3)] = df_group.reset_index(drop=True)
        
    class BHAC15:
        @staticmethod
        def parse_file(file_path):
            data = {}
            current_key = None
            current_data = []
            columns_line = None
    
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
    
                    if line.startswith('!  t (Gyr) ='):
                        if current_key is not None:
                            columns = ['M/Ms', 'Teff', 'L/Ls', 'g', 'R/Rs', 'Li/Li0', 'F33', 'F33B', 'F41', 'F45B', 'F47', 
                                       'F51', 'FHa', 'F57', 'F63B', 'F67', 'F75', 'F78', 'F82', 'F82B', 'F89', 'G_RSV', 'G', 
                                       'G_BP', 'G_RP']
                            df = pd.DataFrame(current_data, columns=columns)
                            df = df.dropna()
                            df['age(Gyr)'] = float(current_key)
                            data[current_key] = df
    
                        current_key = line.split('=')[1].strip()
                        current_data = []
                    elif line.startswith('!'):
                        if columns_line is None:
                            columns_line = line
                    else:
                        current_data.append(line.split())
    
            if current_key is not None:
                columns = ['M/Ms', 'Teff', 'L/Ls', 'g', 'R/Rs', 'Li/Li0', 'F33', 'F33B', 'F41', 'F45B', 'F47', 
                           'F51', 'FHa', 'F57', 'F63B', 'F67', 'F75', 'F78', 'F82', 'F82B', 'F89', 'G_RSV', 'G', 
                           'G_BP', 'G_RP']
                df = pd.DataFrame(current_data, columns=columns)
                df = df.dropna()
                df['age(Gyr)'] = float(current_key)
                data[current_key] = df
    
            return data
    
        @staticmethod
        def save_to_csv(data_dict, file_path):
            combined_df = pd.concat(data_dict.values(), ignore_index=True)
    
            csv_file_path = file_path.replace('.txt', '.csv')
            combined_df.to_csv(csv_file_path, header=False, index=False)
    
            with open(csv_file_path, 'r') as f:
                lines = f.readlines()
            lines.insert(0, ','.join(combined_df.columns) + '\n')
            with open(csv_file_path, 'w') as f:
                f.writelines(lines)
    
class PlotAnalyzer:
    """
    Module to analyze SPOTS isochrones as a mixture in f factor values, with inference in data points for fixed age.
    For now, just 120 Myr Pleiades data are available, but could be useful for new data for different clusters.
    """
    def __init__(self, path_all):
        self.spots_ybc_instance = Models.SPOTS_YBC(path_all)
        self.btsettl_instance = Models.BTSettl(path_all + 'data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv')
        
        self.SPOTS_edr3 = self.spots_ybc_instance.SPOTS_edr3
        self.BTSettl_Li_isochrones = self.btsettl_instance.BTSettl_Li_isochrones

    
    def perpendicular_distance(self, point, line_start, line_end):
        """
        Distance to line segment from a point, useful to approx. an isochrone for a mixture one.

        """
        line_vec = line_end - line_start
        point_vec = point - line_start
        projection = np.dot(point_vec, line_vec) / np.maximum(np.dot(line_vec, line_vec), 1e-10)  # Projection of point onto the line
        if projection < 0:
            return np.linalg.norm(point - line_start)  # Closest point is the start of the line segment
        elif projection > 1:
            return np.linalg.norm(point - line_end)  # Closest point is the end of the line segment
        else:
            closest_point = line_start + projection * line_vec
            return np.linalg.norm(point - closest_point)  # Closest point is on the line segment

    def find_closest_point_to_line(self, data_point1, data_point2, models, threshold_factor=0.5):
        """
        Closest point to a series of data points from a model isochrone with 2 consecutives points.
        Threshold_factor for perpendicular distance as a minimum.
        
        Returns:
        closest_point and index corresponding to closest_point in model isochrones.

        """
        closest_distance = float('inf')
        closest_point = None
        closest_model_index = None
        for i, model in enumerate(models):
            distances = [self.perpendicular_distance(model_point, data_point1, data_point2) for model_point in model]
            min_distance = np.min(distances)
            if min_distance < closest_distance:
                closest_distance = min_distance
                closest_point = model[np.argmin(distances)]
                closest_model_index = i
        threshold_distance = threshold_factor * self.perpendicular_distance(data_point1, data_point2, data_point2)
        if closest_distance > threshold_distance:
            return None, None
        return closest_point, closest_model_index

    def run_bayesian_inference(self, coord_data, coord, bar=False):
        """
        Bayesian models without outliers for x data obs. point, as a Gaussian distribution of points in y axis.

        """
        # Define priors
        data = coord_data

        mean = data.mean()
        std = data.std()

        with pm.Model() as model:
            # Priors
            sigma = pm.HalfNormal('sigma_'+coord, sigma=std)
            coord_mean = pm.Normal('coord_mean_'+coord, mu=mean, sigma=sigma)

            # Likelihood
            likelihood = pm.Normal('likelihood_'+coord, mu=coord_mean, sigma=sigma, observed=coord_data)

            # Sample from the posterior
            trace = pm.sample(1000, tune=1000, target_accept=0.99, progressbar=bar)

        # Return posterior distribution samples
        return trace, std

    def run_bayesian_inference_out(self, coord_data, coord, bar=False):
        """
        Bayesian models with outliers for y data obs. point, as a Gaussian distribution of points in x axis.
        Outliers such as binary stars or activity in obs. data.

        """
        # Define priors
        data = coord_data

        mean = data.mean()
        std = data.std()

        distance = np.abs(data - mean) / std

        outlier_index = distance.idxmax()

        outlier_value = data[outlier_index]

        data_filtered = data[distance <= 1]

        std_out = data_filtered.std()

        with pm.Model() as model:
            # Priors
            coord_mean = pm.Normal('coord_mean_'+coord, mu=mean, sigma=std_out)

            # Priors for outliers
            P_b = pm.Beta('P_b_'+coord, alpha=2, beta=5)
            Y_b = pm.Normal('Y_b_'+coord, mu=outlier_value, sigma=std)  # Mean and standard deviation of outlier distribution

            # Main and outlier distributions
            like = pm.Normal.dist(mu=coord_mean, sigma=std_out)
            outlier = pm.Normal.dist(mu=Y_b, sigma=np.sqrt(std**2 + std_out**2))

            # Likelihood
            y_likelihood = pm.Mixture('likelihood_'+coord, w=[1 - P_b, P_b], comp_dists=[like, outlier], observed=data)

            # Sample from the posterior
            trace = pm.sample(1000, tune=1000, target_accept=0.99, progressbar=bar)

            return trace, std_out

    def find_closest_age(self, age_array, age):
        closest_age = min(age_array, key=lambda x: abs(x - age))
        return closest_age

    def get_color_gradient(self, num_steps, step):
        blue_value = int(255 * (step / (num_steps - 1)))
        return (0, 0, blue_value / 255)

    def plot_process_CMD(self, SPOTS, data_obs, band_1, band_2, band_y, age_iso, max_mag):
        """
        Function to draw CMD for different mag vs color and interpolated line from data.
        
        Args:
        -band_1: photometric band for first mag in color index.
        -band_2: photometric band for segund mag in color index.
        -band_y: photometric band for y axis.
        -age_iso: isochrone age (for now just 120 Myr) [Gyr]
        -max_mag: maximum in color index in UCDs region.
        
        Returns:
        -intervals: number of intervals in inference process.
        -intervals_x: intervals between points.
        -x, y, e_x, e_y: x, y median values of posterior distributions with sigma.
            
        """
        
        plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
        plt.rcParams['font.family'] = 'serif'
        
        age_array = SPOTS['00'].keys()
        age = self.find_closest_age(age_array, age_iso)
        start_time = time.time()
        interval_x = np.linspace(np.min(SPOTS['00'][age][band_1]-SPOTS['00'][age][band_2]), max_mag, len(SPOTS['85'][age][band_y])-1)
        interval_y = np.linspace(np.min(SPOTS['00'][age][band_y]), np.max(data_obs[band_y]), len(SPOTS['85'][age][band_y])-1)
        self.interval_size_x = np.mean(np.diff(interval_x))
        self.interval_size_y = np.mean(np.diff(interval_y))
        length_x = len(interval_x)
        data_obs_x = {}
        data_obs_y = {}
        for i in range(len(interval_x) - 1):
            mask = (data_obs[band_1]-data_obs[band_2] >= interval_x[i]) & (data_obs[band_1]-data_obs[band_2] < interval_x[i+1])
            selected_data_x = data_obs[band_1][mask]-data_obs[band_2][mask]
            selected_data_y = data_obs[band_y][mask]
            data_obs_x[i] = selected_data_x
            data_obs_y[i] = selected_data_y
        posterior_x = []
        std_array = []
        for i in range(len(interval_x) - 1):
            logging.disable(logging.CRITICAL)
            x_interval_data = data_obs_x[i]
            x_posterior, std = self.run_bayesian_inference(x_interval_data, 'x')
            std_array.append(std)
            posterior_x.append(x_posterior)
            logging.disable(logging.NOTSET)
        posterior_y = []
        std_out_array = []
        for i in range(len(interval_x) - 1):
            logging.disable(logging.CRITICAL)
            y_interval_data = data_obs_y[i]
            y_posterior, std_out = self.run_bayesian_inference_out(y_interval_data, 'y')
            std_out_array.append(std_out)
            posterior_y.append(y_posterior)
            logging.disable(logging.NOTSET)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylabel(f'${band_y.split("_")[0]}$ [mag]')
        ax.set_xlabel(f'${band_1.split("_")[0]}-{band_2.split("_")[0]}$ [mag]')
        step = 0
        for f in ['00', '17', '34', '51', '68', '85']:
            num_steps = len(SPOTS)
            age_Myr = age*1000
            ax.plot(self.SPOTS_edr3[f][age][band_1] - self.SPOTS_edr3[f][age][band_2], self.SPOTS_edr3[f][age][band_y], label='SPOTS-YBC f0'+f+f'; {age_Myr} Myr', color=self.get_color_gradient(num_steps, step), linewidth=1, linestyle='--')
            step = step + 1
        ax.scatter(data_obs[band_1]-data_obs[band_2], data_obs[band_y], s=10, zorder=0, color='r', alpha=0.125)
        for x in interval_x:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, zorder=0)
        x_median_array = []
        y_median_array = []
        e_x_median_array = []
        e_y_median_array = []
        for i in range(len(posterior_x)):
            x_median = np.median(posterior_x[i].posterior['coord_mean_x'].values)
            y_median = np.median(posterior_y[i].posterior['coord_mean_y'].values)
            e_x_median = np.mean(std_array[i])
            e_y_median = np.mean(std_out_array[i])
            x_median_array.append(x_median)
            y_median_array.append(y_median)
            e_x_median_array.append(e_x_median)
            e_y_median_array.append(e_y_median)
        ax.plot(x_median_array, y_median_array, linewidth=1, linestyle='-', color='k', label='Interpolated Line (Bayesian)', zorder=5)
        ax.fill_between(np.sort(x_median_array), np.sort(np.array(y_median_array) + np.array(e_y_median_array)), np.sort(np.array(y_median_array) - np.array(e_y_median_array)), color='yellow', alpha=0.35, zorder=4, label=r'$\sigma_G$')
        ax.fill_between(np.sort(x_median_array), np.sort(np.array(y_median_array) + 3*np.array(e_y_median_array)), np.sort(np.array(y_median_array) - 3*np.array(e_y_median_array)), color='orange', alpha=0.25, zorder=3, label=r'$3\sigma_G$')
        ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(1.25, 0.9))
        ax.invert_yaxis()
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f'Exe. time: {minutes} minutos y {seconds:.2f} segundos.')
        return length_x, interval_x, x_median_array, y_median_array, e_x_median_array, e_y_median_array

    def plot_result(self, interval_x, x_median_array, y_median_array, e_x_median_array, e_y_median_array, SPOTS, band_1, band_2, band_y, data_obs, age_iso, max_mag, l=2, BTSettl=False):
        """
        Function to draw CMD for different mag vs color and mixture isochrone in f factor, as inference result interpolation.
        
        Args:
        -intervals: number of intervals in inference process.
        -intervals_x: intervals between points.
        -x, y, e_x, e_y: x, y median values of posterior distributions with sigma.
        
        Returns:
        -SPOTS_ISO: SPOTS mixture isochrone.
        
        """
        
        plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
        plt.rcParams['font.family'] = 'serif'
        
        age_array = SPOTS['00'].keys()
        age = self.find_closest_age(age_array, age_iso)
        start_time = time.time()
        models = [np.column_stack((SPOTS['00'][age][band_1]-SPOTS['00'][age][band_2], SPOTS['00'][age][band_y])),
                 np.column_stack((SPOTS['17'][age][band_1]-SPOTS['17'][age][band_2], SPOTS['17'][age][band_y])),
                 np.column_stack((SPOTS['34'][age][band_1]-SPOTS['34'][age][band_2], SPOTS['34'][age][band_y])),
                 np.column_stack((SPOTS['51'][age][band_1]-SPOTS['51'][age][band_2], SPOTS['51'][age][band_y])),
                 np.column_stack((SPOTS['68'][age][band_1]-SPOTS['68'][age][band_2], SPOTS['68'][age][band_y])),
                 np.column_stack((SPOTS['85'][age][band_1]-SPOTS['85'][age][band_2], SPOTS['85'][age][band_y]))]
        models_info = [            np.column_stack([SPOTS['00'][age][column] for column in SPOTS['00'][age].columns]),
            np.column_stack([SPOTS['17'][age][column] for column in SPOTS['17'][age].columns]),
            np.column_stack([SPOTS['34'][age][column] for column in SPOTS['34'][age].columns]),
            np.column_stack([SPOTS['51'][age][column] for column in SPOTS['51'][age].columns]),
            np.column_stack([SPOTS['68'][age][column] for column in SPOTS['68'][age].columns]),
            np.column_stack([SPOTS['85'][age][column] for column in SPOTS['85'][age].columns])
        ]
        data = np.column_stack((x_median_array, y_median_array))
        matched_points = []
        model_indexes = []
        data_indexes = []
        closest_point = models[0][-1]
        matched_points.append(closest_point)
        model_indexes.append(0)
        data_indexes.append(len(models[0]) - 1)
        for i in range(1, len(data) - 2):
            data_point1 = data[i - 1]
            data_point2 = data[i + 1]
            closest_point, closest_model_index = self.find_closest_point_to_line(data_point1, data_point2, models)
            if np.array_equal(closest_point, matched_points[-1]):
                continue
            if closest_point[0] < matched_points[-1][0]:
                matched_points.insert(-1, closest_point)
                model_indexes.insert(-1, closest_model_index)
                closest_model = models[closest_model_index]
                closest_point_index = np.argmin([self.perpendicular_distance(model_point, data_point1, data_point2) for model_point in closest_model])
                data_index = closest_point_index
                data_indexes.insert(-1, data_index)
            else:
                matched_points.append(closest_point)
                model_indexes.append(closest_model_index)
                closest_model = models[closest_model_index]
                closest_point_index = np.argmin([self.perpendicular_distance(model_point, data_point1, data_point2) for model_point in closest_model])
                data_index = closest_point_index
                data_indexes.append(data_index)
        self.last_model_index = len(matched_points) - 1
        for i in range(l):
            closest_point = models[model_indexes[-1]][l-1-i]
            matched_points.append(closest_point)
            model_indexes.append(model_indexes[-1])
            data_indexes.append(l-1-i)
        final_iso_array = np.array(matched_points)
        final_iso_data = []
        for model_index, data_index in zip(model_indexes, data_indexes):
            row_data = models_info[model_index][data_index].tolist()
            row_data.append([0, 17, 34, 51, 68, 85][model_index])
            final_iso_data.append(row_data)
        column_names = list(SPOTS['00'][age].columns) + ['f(%)']
        final_iso_df = pd.DataFrame(final_iso_data, columns=column_names)
        SPOTS_iso = {age: final_iso_df}
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylabel(f'${band_y.split("_")[0]}$ [mag]')
        ax.set_xlabel(f'${band_1.split("_")[0]}-{band_2.split("_")[0]}$ [mag]')
        step = 0
        for f in ['00', '17', '34', '51', '68', '85']:
            num_steps = len(SPOTS)
            age_Myr = age*1000
            ax.plot(self.SPOTS_edr3[f][age][band_1] - self.SPOTS_edr3[f][age][band_2], self.SPOTS_edr3[f][age][band_y], label='SPOTS-YBC f0'+f+f'; {age_Myr} Myr', color=self.get_color_gradient(num_steps, step), linewidth=1, linestyle='--')
            step = step + 1
        ax.scatter(data_obs[band_1]-data_obs[band_2], data_obs[band_y], s=10, zorder=0, color='r', alpha=0.125)
        for x in interval_x:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, zorder=0)
        ax.plot(x_median_array, y_median_array, linewidth=1, linestyle=':', color='k', label='Interpolated Isochrone (Bayesian)', zorder=5)
        ax.plot(final_iso_array[:,0], final_iso_array[:,1], linewidth=1, linestyle='-', color='k', label='Mixture Isochrone', zorder=5)
        ax.fill_between(np.sort(x_median_array), np.sort(np.array(y_median_array) + np.array(e_y_median_array)), np.sort(np.array(y_median_array) - np.array(e_y_median_array)), color='yellow', alpha=0.35, zorder=4, label=r'$\sigma_G$')
        ax.fill_between(np.sort(x_median_array), np.sort(np.array(y_median_array) + 3*np.array(e_y_median_array)), np.sort(np.array(y_median_array) - 3*np.array(e_y_median_array)), color='orange', alpha=0.25, zorder=3, label=r'$3\sigma_G$')
        if BTSettl == True:
            age_nearest = select_nearest_age(self.BTSettl_Li_isochrones, age)
            BTSettl_Li_isochrones_Teff = self.BTSettl_Li_isochrones[age_nearest][self.BTSettl_Li_isochrones[age_nearest]['Teff'] < 2955]
            BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff[BTSettl_Li_isochrones_Teff['Teff'] > 1600]
            ax.scatter(BTSettl_Li_isochrones_Teff[band_1] - BTSettl_Li_isochrones_Teff[band_2], BTSettl_Li_isochrones_Teff[band_y], s=10, label='Low-mass stars BT-Settl')
        else:
            pass
        ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(1.35, 0.9))
        ax.invert_yaxis()
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f'Exe. time: {minutes} minutos y {seconds:.2f} segundos.')
        return SPOTS_iso
    
    def plot_process_HRD(self, SPOTS, data_obs, age_iso):
        """
        Function to draw HDR and interpolated line from data.
        
        Args:
        ... in Teff.
        -age_iso: isochrone age (for now just 120 Myr) [Gyr]
        
        Returns:
        -intervals: number of intervals in inference process.
        -intervals_x: intervals between points.
        -x, y, e_x, e_y: x, y median values of posterior distributions with sigma.
        ... in Teff.  
        
        """
        
        plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
        plt.rcParams['font.family'] = 'serif'
        
        age_array = SPOTS['00'].keys()
        age = self.find_closest_age(age_array, age_iso)
        start_time = time.time()
        interval_x_Teff = np.linspace(np.min(SPOTS['00'][age]['Teff']), 6600, len(SPOTS['85'][age]['Teff'])-1)
        interval_y_Teff = np.linspace(np.min(SPOTS['00'][age]['log(L/Lsun)']), np.max(data_obs['log(L/Lsun)']), len(SPOTS['85'][age]['log(L/Lsun)'])-1)
        self.interval_size_x_Teff = np.mean(np.diff(interval_x_Teff))
        self.interval_size_y_Teff = np.mean(np.diff(interval_y_Teff))
        data_obs_x_Teff = {}
        data_obs_y_Teff = {}
        length_x_Teff = len(interval_x_Teff)
        # Iterate over the intervals defined by interval_x
        for i in range(len(interval_x_Teff) - 1):
            # Select data within the current interval
            mask = (data_obs['Teff_x'] >= interval_x_Teff[i]) & (data_obs['Teff_x'] < interval_x_Teff[i+1])
            selected_data_x_Teff = data_obs['Teff_x'][mask]
            selected_data_y_Teff = data_obs['log(L/Lsun)'][mask]
            
            # Add the selected data to data_obs_x with the corresponding index as key
            data_obs_x_Teff[i] = selected_data_x_Teff
            data_obs_y_Teff[i] = selected_data_y_Teff
        posterior_x_Teff = []
        std_array_Teff = []
        for i in range(len(interval_x_Teff) - 1):
            logging.disable(logging.CRITICAL)
            x_interval_data = data_obs_x_Teff[i]
            x_posterior, std = self.run_bayesian_inference(x_interval_data, 'x')
            std_array_Teff.append(std)
            posterior_x_Teff.append(x_posterior)
            logging.disable(logging.NOTSET)
        posterior_y_Teff = []
        std_out_array_Teff = []
        for i in range(len(interval_x_Teff) - 1):
            logging.disable(logging.CRITICAL)
            y_interval_data = data_obs_y_Teff[i]
            y_posterior, std_out = self.run_bayesian_inference_out(y_interval_data, 'y')
            std_out_array_Teff.append(std_out)
            posterior_y_Teff.append(y_posterior)
            logging.disable(logging.NOTSET)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylabel('$\log{(\mathcal{L}/\mathcal{L}_{\odot})}$')
        ax.set_xlabel('$T_{eff}$ [K]')
        step = 0
        for f in ['00', '17', '34', '51', '68', '85']:
            num_steps = len(SPOTS)
            age_Myr = age*1000
            ax.plot(self.SPOTS_edr3[f][age]['Teff'], self.SPOTS_edr3[f][age]['log(L/Lsun)'], label='SPOTS-YBC f0'+f+f'; {age_Myr} Myr', color=self.get_color_gradient(num_steps, step), linewidth=1, linestyle='--')
            step = step + 1
        ax.scatter(data_obs['Teff_x'], data_obs['log(L/Lsun)'], s=10, zorder=0, color='r', alpha=0.125)
        for x in interval_x_Teff:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, zorder=0)
        x_median_array_Teff = []
        y_median_array_Teff = []
        e_x_median_array_Teff = []
        e_y_median_array_Teff = []
        for i in range(len(posterior_x_Teff)):
            x_median_Teff = np.median(posterior_x_Teff[i].posterior['coord_mean_x'].values)
            y_median_Teff = np.median(posterior_y_Teff[i].posterior['coord_mean_y'].values)
            e_x_median_Teff = np.mean(std_array_Teff[i])
            e_y_median_Teff = np.mean(std_out_array_Teff[i])
            x_median_array_Teff.append(x_median_Teff)
            y_median_array_Teff.append(y_median_Teff)
            e_x_median_array_Teff.append(e_x_median_Teff)
            e_y_median_array_Teff.append(e_y_median_Teff)
        ax.plot(x_median_array_Teff, y_median_array_Teff, linewidth=1, linestyle='-', color='k', label='Interpolated Line (Bayesian)', zorder=5)
        ax.fill_between(np.sort(x_median_array_Teff), np.sort(np.array(y_median_array_Teff) + np.array(e_y_median_array_Teff)), np.sort(np.array(y_median_array_Teff) - np.array(e_y_median_array_Teff)), color='yellow', alpha=0.35, zorder=4, label=r'$\sigma_G$')
        ax.fill_between(np.sort(x_median_array_Teff), np.sort(np.array(y_median_array_Teff) + 3*np.array(e_y_median_array_Teff)), np.sort(np.array(y_median_array_Teff) - 3*np.array(e_y_median_array_Teff)), color='orange', alpha=0.25, zorder=3, label=r'$3\sigma_G$')
        ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(1.25, 0.9))
        ax.set_xlim(1500, 7000)
        ax.invert_xaxis()
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f'Exe. time: {minutes} minutos y {seconds:.2f} segundos.')
        return length_x_Teff, interval_x_Teff, x_median_array_Teff, y_median_array_Teff, e_x_median_array_Teff, e_y_median_array_Teff

    def plot_result_HRD(self, interval_x_Teff, x_median_array_Teff, y_median_array_Teff, e_x_median_array_Teff, e_y_median_array_Teff, SPOTS, data_obs, age_iso, l=2, BTSettl=False):
        """
        Function to draw CMD for different mag vs color and mixture isochrone in f factor, as inference result interpolation.
        
        Args:
        ... in Teff.
        
        Returns:
        -SPOTS_ISO: SPOTS mixture isochrone in Teff.
        
        """
        
        plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
        plt.rcParams['font.family'] = 'serif'
        
        age_array = SPOTS['00'].keys()
        age = self.find_closest_age(age_array, age_iso)
        start_time = time.time()
        models = [np.column_stack((SPOTS['00'][age]['Teff'], SPOTS['00'][age]['log(L/Lsun)'])),
                 np.column_stack((SPOTS['17'][age]['Teff'], SPOTS['17'][age]['log(L/Lsun)'])),
                 np.column_stack((SPOTS['34'][age]['Teff'], SPOTS['34'][age]['log(L/Lsun)'])),
                 np.column_stack((SPOTS['51'][age]['Teff'], SPOTS['51'][age]['log(L/Lsun)'])),
                 np.column_stack((SPOTS['68'][age]['Teff'], SPOTS['68'][age]['log(L/Lsun)'])),
                 np.column_stack((SPOTS['85'][age]['Teff'], SPOTS['85'][age]['log(L/Lsun)']))]
        models_info = [            np.column_stack([SPOTS['00'][age][column] for column in SPOTS['00'][age].columns]),
            np.column_stack([SPOTS['17'][age][column] for column in SPOTS['17'][age].columns]),
            np.column_stack([SPOTS['34'][age][column] for column in SPOTS['34'][age].columns]),
            np.column_stack([SPOTS['51'][age][column] for column in SPOTS['51'][age].columns]),
            np.column_stack([SPOTS['68'][age][column] for column in SPOTS['68'][age].columns]),
            np.column_stack([SPOTS['85'][age][column] for column in SPOTS['85'][age].columns])
        ]
        data = np.column_stack((x_median_array_Teff, y_median_array_Teff))
        matched_points = []
        model_indexes = []
        data_indexes = []
        closest_point = models[0][-1]
        matched_points.append(closest_point)
        model_indexes.append(0)
        data_indexes.append(len(models[0]) - 1)
        for i in range(1, len(data) - 2):
            data_point1 = data[i - 1]
            data_point2 = data[i + 1]
            closest_point, closest_model_index = self.find_closest_point_to_line(data_point1, data_point2, models)
            if np.array_equal(closest_point, matched_points[-1]):
                continue
            if closest_point[0] < matched_points[-1][0]:
                matched_points.insert(-1, closest_point)
                model_indexes.insert(-1, closest_model_index)
                closest_model = models[closest_model_index]
                closest_point_index = np.argmin([self.perpendicular_distance(model_point, data_point1, data_point2) for model_point in closest_model])
                data_index = closest_point_index
                data_indexes.insert(-1, data_index)
            else:
                matched_points.append(closest_point)
                model_indexes.append(closest_model_index)
                closest_model = models[closest_model_index]
                closest_point_index = np.argmin([self.perpendicular_distance(model_point, data_point1, data_point2) for model_point in closest_model])
                data_index = closest_point_index
                data_indexes.append(data_index)
        self.last_model_index = len(matched_points) - 1
        final_iso_array = np.array(matched_points)
        final_iso_data = []
        for model_index, data_index in zip(model_indexes, data_indexes):
            row_data = models_info[model_index][data_index].tolist()
            row_data.append([0, 17, 34, 51, 68, 85][model_index])
            final_iso_data.append(row_data)
        column_names = list(SPOTS['00'][age].columns) + ['f(%)']
        final_iso_df = pd.DataFrame(final_iso_data, columns=column_names)
        SPOTS_iso_Teff = {age: final_iso_df}
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylabel('$\log{(\mathcal{L}/\mathcal{L}_{\odot})}$')
        ax.set_xlabel('$T_{eff}$ [K]')
        step = 0
        for f in ['00', '17', '34', '51', '68', '85']:
            num_steps = len(SPOTS)
            age_Myr = age*1000
            ax.plot(self.SPOTS_edr3[f][age]['Teff'], self.SPOTS_edr3[f][age]['log(L/Lsun)'], label='SPOTS-YBC f0'+f+f'; {age_Myr} Myr', color=self.get_color_gradient(num_steps, step), linewidth=1, linestyle='--')
            step = step + 1
        ax.scatter(data_obs['Teff_x'], data_obs['log(L/Lsun)'], s=10, zorder=0, color='r', alpha=0.125)
        for x in interval_x_Teff:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, zorder=0)
        ax.plot(x_median_array_Teff, y_median_array_Teff, linewidth=1, linestyle=':', color='k', label='Interpolated Isochrone (Bayesian)', zorder=5)
        ax.plot(final_iso_array[:,0], final_iso_array[:,1], linewidth=1, linestyle='-', color='k', label='Mixture Isochrone', zorder=5)
        ax.fill_between(np.sort(x_median_array_Teff), np.sort(np.array(y_median_array_Teff) + np.array(e_y_median_array_Teff)), np.sort(np.array(y_median_array_Teff) - np.array(e_y_median_array_Teff)), color='yellow', alpha=0.35, zorder=4, label=r'$\sigma_G$')
        ax.fill_between(np.sort(x_median_array_Teff), np.sort(np.array(y_median_array_Teff) + 3*np.array(e_y_median_array_Teff)), np.sort(np.array(y_median_array_Teff) - 3*np.array(e_y_median_array_Teff)), color='orange', alpha=0.25, zorder=3, label=r'$3\sigma_G$')
        if BTSettl == True:
            age_nearest = select_nearest_age(self.BTSettl_Li_isochrones, age)
            BTSettl_Li_isochrones_Teff = self.BTSettl_Li_isochrones[age_nearest][self.BTSettl_Li_isochrones[age_nearest]['Teff'] < 2955]
            BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff[BTSettl_Li_isochrones_Teff['Teff'] > 1600]
            ax.scatter(BTSettl_Li_isochrones_Teff['Teff'], BTSettl_Li_isochrones_Teff['log(L/Lsun)'], s=10, label='Low-mass stars BT-Settl')
        else:
            pass
        ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(1.35, 0.9))
        ax.set_xlim(1500, 7000)
        ax.invert_xaxis()
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f'Exe. time: {minutes} minutos y {seconds:.2f} segundos.')
        return SPOTS_iso_Teff


    def plot_HRD(self, SPOTS, data_obs, age_iso, f_FGK, f_UCDs, mid):
        """
        Plot HRD without inference interpolation.
        
        mid value must be test in plots!

        """
        
        plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
        plt.rcParams['font.family'] = 'serif'
        
        
        if type(f_FGK) != str or type(f_UCDs) != str:
            f_FGK = f'{f_FGK}'
            f_UCDs = f'{f_UCDs}'
        else:
            pass
        
        age_array = SPOTS['00'].keys()
        age = self.find_closest_age(age_array, age_iso)
        
        interval_x = np.linspace(np.min(SPOTS['00'][age]['Teff']), 6600, len(SPOTS['85'][age]['Teff'])-1)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylabel('$\log{(\mathcal{L}/\mathcal{L}_{\odot})}$')
        ax.set_xlabel('$T_{eff}$ [K]')
        step = 0
        for f in ['00', '17', '34', '51', '68', '85']:
            num_steps = len(SPOTS)
            age_Myr = age*1000
            ax.plot(self.SPOTS_edr3[f][age]['Teff'], self.SPOTS_edr3[f][age]['log(L/Lsun)'], label='SPOTS-YBC f0'+f+f'; {age_Myr} Myr', color=self.get_color_gradient(num_steps, step), linewidth=1, linestyle='--')
            step = step + 1
        ax.scatter(data_obs['Teff_x'], data_obs['log(L/Lsun)'], s=10, zorder=0, color='r', alpha=0.125)
        
        data_obs_x_HRD = {}
        data_obs_y_HRD = {}

        # Iterate over the intervals defined by interval_x
        for i in range(len(interval_x) - 1):
            # Select data within the current interval
            mask = (data_obs['Teff_x'] >= interval_x[i]) & (data_obs['Teff_x'] < interval_x[i+1])
            selected_data_x = data_obs['Teff_x'][mask]
            selected_data_y = data_obs['log(L/Lsun)'][mask]
            
            # Add the selected data to data_obs_x with the corresponding index as key
            data_obs_x_HRD[i] = selected_data_x
            data_obs_y_HRD[i] = selected_data_y
        
        SPOTS_Teff_iso = []
        SPOTS_logL_iso = []
    
        for i in range(1, mid):
            SPOTS_Teff_iso.append(SPOTS[f_FGK][age]['Teff'].values[-i])
            SPOTS_logL_iso.append(SPOTS[f_FGK][age]['log(L/Lsun)'].values[-i])
            
        for i in range(mid, len(SPOTS[f_UCDs][age]['Teff'])):
            SPOTS_Teff_iso.append(SPOTS[f_UCDs][age]['Teff'].values[-i])
            SPOTS_logL_iso.append(SPOTS[f_UCDs][age]['log(L/Lsun)'].values[-i])
        
        ax.plot(SPOTS_Teff_iso, SPOTS_logL_iso, zorder=5, label='Mixture Isochrone', color='k', linewidth=1)
        
        ax.scatter(data_obs['Teff_x'], data_obs['log(L/Lsun)'], s=10, zorder=0, color='r', alpha=0.125)
        
        age_nearest = select_nearest_age(self.BTSettl_Li_isochrones, age)
        BTSettl_Li_isochrones_Teff = self.BTSettl_Li_isochrones[age_nearest][self.BTSettl_Li_isochrones[age_nearest]['Teff'] < 2955]
        BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff[BTSettl_Li_isochrones_Teff['Teff'] > 1600]
        ax.scatter(BTSettl_Li_isochrones_Teff['Teff'], BTSettl_Li_isochrones_Teff['log(L/Lsun)'], s=10, label='Low-mass stars BT-Settl')
        
        ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(1.25, 0.9))
        
        ax.set_xlim(2000, 7000)
        
        ax.invert_xaxis()
        
        return {age: pd.DataFrame({'Teff': SPOTS_Teff_iso, 'log(L/Lsun)': SPOTS_logL_iso})}
    
    def plot_ages(self, SPOTS, f, minim_age, maxim_age, upplim, BTSettl_str=True):
        Teff_max_array_SPOTS = []
        Teff_min_array_SPOTS = []
        age_array_SPOTS = []
    
        if BTSettl_str == True:
            BTSettl = self.BTSettl_Li_isochrones
        else:
            raise ValueError('BTSettl must be True.')
    
        if type(f) != str:
            f = f'{f}'
        else:
            pass
    
        for age in SPOTS[f].keys():
            age_array_SPOTS.append(age)
            Teff_max = max(SPOTS[f][age]['Teff'])
            Teff_min = min(SPOTS[f][age]['Teff'])
            Teff_max_array_SPOTS.append(Teff_max)
            Teff_min_array_SPOTS.append(Teff_min)
    
        Teff_max_array_BTSettl = []
        Teff_min_array_BTSettl = []
        age_array_BTSettl = []
    
        for age in BTSettl.keys():
            age_array_BTSettl.append(age)
            Teff_max = max(BTSettl[age]['Teff'])
            Teff_min = min(BTSettl[age]['Teff'])
            Teff_max_array_BTSettl.append(Teff_max)
            Teff_min_array_BTSettl.append(Teff_min)
    
        plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
        plt.rcParams['font.family'] = 'serif'
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    
        ax1.scatter(Teff_min_array_SPOTS, age_array_SPOTS, s=10, c='r', label=f'$T_{{eff,\,min}}$ SPOTS-YBC $f=0.{f}$', zorder=2)
    
        ax1.scatter(Teff_min_array_BTSettl, age_array_BTSettl, s=10, c='magenta', label=r'$T_{eff,\,min}$ BT-Settl', zorder=2)
    
        ax1.plot(Teff_min_array_SPOTS, age_array_SPOTS, alpha=0.25, c='r', zorder=1)
    
        ax1.plot(Teff_min_array_BTSettl, age_array_BTSettl, alpha=0.25, c='magenta', zorder=1)
    
        ax1.set_ylabel('Age [Gyr]')
    
        ax1.set_title(r'$T_{eff}$ range')
    
        ax1.set_ylim(-0.025, upplim)
    
        ax1.axhline(maxim_age, lw=1, ls='-.', c='k', zorder=0, alpha=0.25)
        ax1.axhline(minim_age, lw=1, ls='--', c='k', zorder=0, alpha=0.25)
    
        ax2.scatter(Teff_max_array_SPOTS, age_array_SPOTS, s=10, c='b', label=f'$T_{{eff,\,min}}$ SPOTS-YBC $f=0.{f}$', zorder=2)
    
        ax2.scatter(Teff_max_array_BTSettl, age_array_BTSettl, s=10, c='orange', label=r'$T_{eff,\,max}$ BT-Settl', zorder=2)
    
        ax2.plot(Teff_max_array_SPOTS, age_array_SPOTS, alpha=0.25, c='b', zorder=1)
    
        ax2.plot(Teff_max_array_BTSettl, age_array_BTSettl, alpha=0.25, c='orange', zorder=1)
    
        ax2.set_ylim(-0.025, upplim)
    
        ax2.axhline(maxim_age, lw=1, ls='-.', c='k', zorder=0, alpha=0.25, label='120 Myr')
        ax2.axhline(minim_age, lw=1, ls='--', c='k', zorder=0, alpha=0.25, label='20 Myr')
    
        ax2.set_ylabel('Age [Gyr]')
    
        ax2.set_xlabel(r'$T_{eff}$ [K]')
    
        fig.legend(bbox_to_anchor=(1.295, 0.9))
    
    
        Teff_max_array_SPOTS = []
        Teff_min_array_SPOTS = []
        age_array_SPOTS = []
    
        Teff_max_array_SPOTS_Li = []
        Teff_min_array_SPOTS_Li = []
        age_array_SPOTS_Li = []
    
        for age in SPOTS[f].keys():
            if age <= 0.6:
                Teff_series = np.array(SPOTS[f][age]['Teff'])
                A_Li_series = np.array(SPOTS[f][age]['A(Li)'])
    
                Teff_min_all = min(Teff_series)
    
                Teff_max = None
    
                for i in range(len(Teff_series)):
                    if A_Li_series[i] == -np.inf:
                        Teff_min = Teff_series[i-1]
                        if i == 0:
                            Teff_min = Teff_min_all
                        else:
                            pass
                        break
    
                for i in range(len(Teff_series)-1, -1, -1):
                    if A_Li_series[i] == -np.inf:
                        Teff_max = Teff_series[i+1]
                        break
    
                if Teff_max is not None:
                    age_array_SPOTS_Li.append(age)
                    Teff_max_array_SPOTS_Li.append(Teff_max)
                    Teff_min_array_SPOTS_Li.append(Teff_min)
    
        for age in SPOTS[f].keys():
            age_array_SPOTS.append(age)
            Teff_max = max(SPOTS[f][age]['Teff'])
            Teff_min = min(SPOTS[f][age]['Teff'])
            Teff_max_array_SPOTS.append(Teff_max)
            Teff_min_array_SPOTS.append(Teff_min)
    
        Teff_max_array_BTSettl_Li = []
        Teff_min_array_BTSettl_Li = []
        age_array_BTSettl_Li = []
    
        for age in BTSettl.keys():
            if age <= 0.6:
                Teff_series = np.array(BTSettl[age]['Teff'])
                A_Li_series = np.array(BTSettl[age]['A(Li)'])
    
                Teff_min = None
    
                Teff_max = None
    
                for i in range(len(Teff_series)):
                    if A_Li_series[i] == -np.inf:
                        Teff_min = Teff_series[i-1]
                        break
    
                for i in range(len(Teff_series)-1, -1, -1):
                    if A_Li_series[i] == -np.inf:
                        Teff_max = Teff_series[i+1]
                        break
    
                if Teff_max is not None:
                    age_array_BTSettl_Li.append(age)
                    Teff_max_array_BTSettl_Li.append(Teff_max)
                    Teff_min_array_BTSettl_Li.append(Teff_min)
    
        Teff_max_array_BTSettl = []
        Teff_min_array_BTSettl = []
        age_array_BTSettl = []
    
        for age in BTSettl.keys():
            age_array_BTSettl.append(age)
            Teff_max = max(BTSettl[age]['Teff'])
            Teff_min = min(BTSettl[age]['Teff'])
            Teff_max_array_BTSettl.append(Teff_max)
            Teff_min_array_BTSettl.append(Teff_min)
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    
        ax1.scatter(Teff_min_array_SPOTS, age_array_SPOTS, s=10, c='r', label=f'$T_{{eff,\,min}}$ SPOTS-YBC $f=0.{f}$', zorder=2)
    
        ax1.scatter(Teff_min_array_BTSettl, age_array_BTSettl, s=10, c='orange', label=r'$T_{eff,\,min}$ BT-Settl', zorder=2)
    
        ax1.plot(Teff_min_array_BTSettl, age_array_BTSettl, alpha=0.25, c='orange', zorder=1)
    
        ax1.plot(Teff_min_array_SPOTS, age_array_SPOTS, alpha=0.25, c='r', zorder=1)
    
    
        ax1.scatter(Teff_min_array_BTSettl_Li, age_array_BTSettl_Li, s=10, marker='>', c='magenta', label=r'$T_{eff,\,min,\,LDB}$ BT-Settl', zorder=2)
    
        ax1.scatter(Teff_min_array_SPOTS_Li, age_array_SPOTS_Li, s=10, marker='>', c='k', label=f'$T_{{eff,\,min,\,LDB}}$ SPOTS-YBC $f={f}$', zorder=2)
    
        ax1.plot(Teff_min_array_BTSettl_Li, age_array_BTSettl_Li, alpha=0.25, c='magenta', zorder=1)
    
        ax1.plot(Teff_min_array_SPOTS_Li, age_array_SPOTS_Li, alpha=0.25, c='k', zorder=1)
    
    
        ax1.scatter(Teff_max_array_BTSettl_Li, age_array_BTSettl_Li, s=10, marker='<', c='magenta', label=r'$T_{eff,\,max,\,LDB}$ BT-Settl', zorder=2)
    
        ax1.scatter(Teff_max_array_SPOTS_Li, age_array_SPOTS_Li, s=10, marker='<', c='k', label=f'$T_{{eff,\,max,\,LDB}}$ SPOTS-YBC $f={f}$', zorder=2)
    
        ax1.plot(Teff_max_array_BTSettl_Li, age_array_BTSettl_Li, alpha=0.25, ls='--', c='magenta', zorder=1)
    
        ax1.plot(Teff_max_array_SPOTS_Li, age_array_SPOTS_Li, alpha=0.25, ls='--', c='k', zorder=1)
    
        ax1.set_xlim(Teff_min_array_SPOTS[0]-500, 5200)
    
        ax1.set_ylabel('Age [Gyr]')
    
        ax1.set_title(r'$T_{eff}$ range')
    
        ax1.set_ylim(-0.025, upplim)
    
        ax1.fill_betweenx(age_array_BTSettl_Li, Teff_min_array_BTSettl_Li, Teff_max_array_BTSettl_Li, color='green', alpha=0.1, label='LDB BT-Settl')
    
        ax1.fill_betweenx(age_array_SPOTS_Li, Teff_min_array_SPOTS_Li, Teff_max_array_SPOTS_Li, color='magenta', alpha=0.1, label='LDB SPOTS')
    
    
        ax1.axhline(maxim_age, lw=1, ls='-.', c='k', zorder=0, alpha=0.25)
        ax1.axhline(minim_age, lw=1, ls='--', c='k', zorder=0, alpha=0.25)
    
        ax2.scatter(Teff_min_array_SPOTS, age_array_SPOTS, s=10, c='r', zorder=2)
    
        ax2.scatter(Teff_min_array_BTSettl, age_array_BTSettl, s=10, c='orange', zorder=2)
    
        ax2.plot(Teff_min_array_BTSettl, age_array_BTSettl, alpha=0.25, c='orange', zorder=1)
    
        ax2.plot(Teff_min_array_SPOTS, age_array_SPOTS, alpha=0.25, c='r', zorder=1)
    
    
        ax2.scatter(Teff_min_array_BTSettl_Li, age_array_BTSettl_Li, s=10, marker='>', c='magenta', zorder=2)
    
        ax2.scatter(Teff_min_array_SPOTS_Li, age_array_SPOTS_Li, s=10, marker='>', c='k', zorder=2)
    
        ax2.plot(Teff_min_array_BTSettl_Li, age_array_BTSettl_Li, alpha=0.25, c='magenta', zorder=1)
    
        ax2.plot(Teff_min_array_SPOTS_Li, age_array_SPOTS_Li, alpha=0.25, c='k', zorder=1)
    
    
        ax2.scatter(Teff_max_array_BTSettl_Li, age_array_BTSettl_Li, s=10, marker='<', c='magenta', zorder=2)
    
        ax2.scatter(Teff_max_array_SPOTS_Li, age_array_SPOTS_Li, s=10, marker='<', c='k', zorder=2)
    
        ax2.plot(Teff_max_array_BTSettl_Li, age_array_BTSettl_Li, alpha=0.25, ls='--', c='magenta', zorder=1)
    
        ax2.plot(Teff_max_array_SPOTS_Li, age_array_SPOTS_Li, alpha=0.25, ls='--', c='k', zorder=1)
    
        ax2.set_xlim(Teff_min_array_SPOTS[0]-500, 3500)
    
        ax2.set_ylabel('Age [Gyr]')
    
        ax2.set_ylim(-0.025, upplim)
    
        ax2.axhline(maxim_age, lw=1, ls='-.', c='k', zorder=0, alpha=0.25, label='120 Myr')
        ax2.axhline(minim_age, lw=1, ls='--', c='k', zorder=0, alpha=0.25, label='20 Myr')
    
    
        fig.legend(bbox_to_anchor=(1.295, 0.9))
    
class SPOTS_extended:
    def __init__(self, MS_color_file, BTSettl, SPOTS, f_values):
        self.MS_color_file = MS_color_file
        self.BTSettl = BTSettl
        self.SPOTS = SPOTS
        self.f_values = f_values

    @staticmethod
    def find_nearest_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    @staticmethod
    def calculate_log_g(mass, radius):
        """
        Calculate the surface gravity log(g) of a star.
        
        Parameters:
        - mass: Mass of the star in solar masses.
        - radius: Radius of the star in solar radii.
        
        Returns:
        - log_g: The surface gravity log(g) in cgs units.
        """
        G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
        M_sun = 1.98847e30  # mass of the Sun in kg
        R_sun = 6.9634e8  # radius of the Sun in meters
        
        # Convert mass and radius to SI units
        mass_kg = mass * M_sun
        radius_m = radius * R_sun
        
        # Calculate g
        g = G * mass_kg / (radius_m ** 2)
        
        # Convert g to cgs units (cm/s^2)
        g_cgs = g * 100  # 1 m/s^2 = 100 cm/s^2
        
        # Calculate log(g)
        log_g = np.log10(g_cgs)
        
        return log_g

    @staticmethod
    def invert_cmap(cmap):
        cmap_listed = cmap(np.linspace(0, 1, cmap.N))
        return cm.colors.ListedColormap(cmap_listed[::-1])


    def SPOTS_extension(self):
        """
        Extend SPOTS data using provided inputs.
        
        Parameters:
            MS_color (DataFrame): DataFrame for Phot. correction (Pecaut & Mamajek 2013)
            BTSettl (dict): Dictionary with BTSettl data.
            SPOTS (dict): Dictionary with SPOTS data.
            f (str): Factor value to use.
        
        Returns:
            SPOTS_expanded
        """
        M_bol_sun = 4.755
        Teff_sun = 5772
        
        SPOTS_expanded = {}
        
        BTSettl_Li_isochrones_Teff_dic = {}
        
        for f in self.SPOTS.keys():
            Teff_max_array_SPOTS = []
            Teff_min_array_SPOTS = []
            age_array_SPOTS = []
    
            Teff_max_array_SPOTS_Li = []
            Teff_min_array_SPOTS_Li = []
            age_array_SPOTS_Li = []
    
            inf_index_SPOTS_array = []
    
            j = 0
    
            for age in self.SPOTS[f].keys():
                age_array_SPOTS.append(age)
                Teff_max = max(self.SPOTS[f][age]['Teff'])
                Teff_min = min(self.SPOTS[f][age]['Teff'])
                Teff_max_array_SPOTS.append(Teff_max)
                Teff_min_array_SPOTS.append(Teff_min)
    
                j += 1
    
                if age <= 0.7:
                    Teff_series = np.array(self.SPOTS[f][age]['Teff'])
                    A_Li_series = np.array(self.SPOTS[f][age]['A(Li)'])
    
                    Teff_min_all = min(Teff_series)
    
                    Teff_max = None
    
                    for i in range(len(Teff_series)):
                        if A_Li_series[i] == -np.inf:
                            Teff_min = Teff_series[i-1]
                            if i == 0:
                                Teff_min = Teff_min_all
                            else:
                                pass
                            inf_index_SPOTS_array.append(j - 1)
                            break
    
                    for i in range(len(Teff_series)-1, -1, -1):
                        if A_Li_series[i] == -np.inf:
                            Teff_max = Teff_series[i]
                            break
    
                    if Teff_max is not None:
                        age_array_SPOTS_Li.append(age)
                        Teff_max_array_SPOTS_Li.append(Teff_max)
                        Teff_min_array_SPOTS_Li.append(Teff_min)
    
            Teff_max_array_BTSettl = []
            Teff_min_array_BTSettl = []
            age_array_BTSettl = []
    
            Teff_max_array_BTSettl_Li = []
            Teff_min_array_BTSettl_Li = []
            age_array_BTSettl_Li = []
    
            for age in self.BTSettl.keys():
                age_array_BTSettl.append(age)
                Teff_max = max(self.BTSettl[age]['Teff'])
                Teff_min = min(self.BTSettl[age]['Teff'])
                Teff_max_array_BTSettl.append(Teff_max)
                Teff_min_array_BTSettl.append(Teff_min)
    
                if age <= 0.7:
                    Teff_series = np.array(self.BTSettl[age]['Teff'])
                    A_Li_series = np.array(self.BTSettl[age]['A(Li)'])
    
                    Teff_min = None
    
                    Teff_max = None
    
                    for i in range(len(Teff_series)):
                        if A_Li_series[i] == -np.inf:
                            Teff_min = Teff_series[i-1]
                            break
    
                    for i in range(len(Teff_series)-1, -1, -1):
                        if A_Li_series[i] == -np.inf:
                            Teff_max = Teff_series[i+1]
                            break
    
                    if Teff_max is not None:
                        age_array_BTSettl_Li.append(age)
                        Teff_max_array_BTSettl_Li.append(Teff_max)
                        Teff_min_array_BTSettl_Li.append(Teff_min)
            
            MS_color = pd.read_csv(self.MS_color_file)
    
            for c in MS_color.columns:
                if c != '#SpT':
                    MS_color[c] = pd.to_numeric(MS_color[c], errors='coerce')
    
            MS_color = MS_color[10**MS_color['logT'] < 6000]
    
            MS_color = MS_color[10**MS_color['logT'] > 1600]
            
            MS_color['G-J'] = MS_color['G-V'] + MS_color['Mv'] - MS_color['M_J']
            MS_color['G-Ks'] = MS_color['G-V'] + MS_color['Mv'] - MS_color['M_Ks']
            MS_color['Teff'] = 10**MS_color['logT']
            MS_color = MS_color.reset_index()
    
            MS_color_filtered = MS_color[(MS_color['Teff'] >= 1750) & (MS_color['Teff'] <= 3000)].copy()
            MS_color_filtered.loc[:, 'logR'] = np.log10(MS_color_filtered['R_Rsun'])
    
            n = MS_color_filtered.shape[0]
    
    
            SPOTS_expanded[f] = {}
    
            age_index = 0
    
            BTSettl_Li_isochrones_Teff_dic[f] = {}
    
            count = 0
    
            for Teff_SPOTS_min in Teff_min_array_SPOTS:
                if inf_index_SPOTS_array[0] >= age_index:
                    age_BTSettl = list(self.BTSettl.keys())[age_index]
                    age_SPOTS = list(self.SPOTS[f].keys())[age_index]
    
                    age_index += 1
    
                    count += 1
    
                    BTSettl_Li_isochrones_Teff = self.BTSettl[age_BTSettl][self.BTSettl[age_BTSettl]['Teff'] < Teff_SPOTS_min]
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff[BTSettl_Li_isochrones_Teff['Teff'] > min(BTSettl_Li_isochrones_Teff['Teff'])]
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff.sort_values(by='Teff', ascending=False)
                    BTSettl_Li_isochrones_Teff['log(Teff)'] = np.log10(BTSettl_Li_isochrones_Teff['Teff'])
                    BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl] = BTSettl_Li_isochrones_Teff.reset_index()
    
                    self.SPOTS[f][age_SPOTS]['log(Teff)'] = np.log10(self.SPOTS[f][age_SPOTS]['Teff'])
                    self.SPOTS[f][age_SPOTS]['M_bol'] = - 10 * np.log10(self.SPOTS[f][age_SPOTS]['Teff']/Teff_sun) - 5 * self.SPOTS[f][age_SPOTS]['log(R/Rsun)'] + M_bol_sun
                    self.SPOTS[f][age_SPOTS] = self.SPOTS[f][age_SPOTS].sort_values(by='log(Teff)', ascending=False)
    
                    SPOTS_expanded[f][age_SPOTS] = self.SPOTS[f][age_SPOTS]
    
                    SPOTS_expanded[f][age_SPOTS]['Lsun'] = 10**SPOTS_expanded[f][age_SPOTS]['log(L/Lsun)']
    
                    SPOTS_expanded[f][age_SPOTS]['M_bol'] = M_bol_sun - 2.5 * SPOTS_expanded[f][age_SPOTS]['log(R/Rsun)']
                    SPOTS_expanded[f][age_SPOTS]['G_abs'] = SPOTS_expanded[f][age_SPOTS]['G-V'] + SPOTS_expanded[f][age_SPOTS]['V_mag']
                    SPOTS_expanded[f][age_SPOTS]['J_abs'] = - SPOTS_expanded[f][age_SPOTS]['G-J'] + SPOTS_expanded[f][age_SPOTS]['G_abs']
                    SPOTS_expanded[f][age_SPOTS]['RP_abs'] = - SPOTS_expanded[f][age_SPOTS]['G-RP'] + SPOTS_expanded[f][age_SPOTS]['G_abs']
                    SPOTS_expanded[f][age_SPOTS]['BP_abs'] = SPOTS_expanded[f][age_SPOTS]['BP-RP'] + SPOTS_expanded[f][age_SPOTS]['RP_abs']
                    SPOTS_expanded[f][age_SPOTS]['H_abs'] = - SPOTS_expanded[f][age_SPOTS]['J-H'] + SPOTS_expanded[f][age_SPOTS]['J_abs']
                    SPOTS_expanded[f][age_SPOTS]['K_abs'] = - SPOTS_expanded[f][age_SPOTS]['G-K'] + SPOTS_expanded[f][age_SPOTS]['G_abs']
    
                    SPOTS_expanded[f][age_SPOTS]['RP-J'] = SPOTS_expanded[f][age_SPOTS]['RP_abs'] - SPOTS_expanded[f][age_SPOTS]['J_abs'] 
                    SPOTS_expanded[f][age_SPOTS]['G-H'] = SPOTS_expanded[f][age_SPOTS]['G_abs'] - SPOTS_expanded[f][age_SPOTS]['H_abs'] 
    
                    SPOTS_expanded[f][age_SPOTS] = SPOTS_expanded[f][age_SPOTS][~pd.isna(SPOTS_expanded[f][age_SPOTS]['A(Li)'])]
    
            for Teff_BTSettl_min_Li, Teff_SPOTS_min in zip(Teff_min_array_BTSettl_Li, Teff_min_array_SPOTS):
                age_BTSettl = list(self.BTSettl.keys())[age_index]
                age_SPOTS = list(self.SPOTS[f].keys())[age_index]
    
    
                if Teff_BTSettl_min_Li <= Teff_SPOTS_min:
    
                    age_index += 1
    
                    count = count + 1
    
                    BTSettl_Li_isochrones_Teff = self.BTSettl[age_BTSettl][self.BTSettl[age_BTSettl]['Teff'] < Teff_SPOTS_min]
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff[BTSettl_Li_isochrones_Teff['Teff'] > min(BTSettl_Li_isochrones_Teff['Teff'])]
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff.sort_values(by='Teff', ascending=False)
                    BTSettl_Li_isochrones_Teff['log(Teff)'] = np.log10(BTSettl_Li_isochrones_Teff['Teff'])
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff.reset_index()
                    BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl] = BTSettl_Li_isochrones_Teff
    
                    MS_color_rows = {}
                    MS_color_rows[age_SPOTS] = pd.DataFrame(np.nan, index=range(n), columns=self.SPOTS[f][age_SPOTS].columns)
    
                    self.SPOTS[f][age_SPOTS]['G-V'] = self.SPOTS[f][age_SPOTS]['G_abs'] - self.SPOTS[f][age_SPOTS]['V_mag']
                    self.SPOTS[f][age_SPOTS]['log(Teff)'] = np.log10(self.SPOTS[f][age_SPOTS]['Teff'])
                    self.SPOTS[f][age_SPOTS]['M_bol'] = - 10 * np.log10(self.SPOTS[f][age_SPOTS]['Teff']/Teff_sun) - 5 * self.SPOTS[f][age_SPOTS]['log(R/Rsun)'] + M_bol_sun
                    self.SPOTS[f][age_SPOTS] = self.SPOTS[f][age_SPOTS].sort_values(by='log(Teff)', ascending=False)
    
                    SPOTS_expanded[f][age_SPOTS] = pd.concat([self.SPOTS[f][age_SPOTS], MS_color_rows[age_SPOTS]], ignore_index=True)
    
                    columns_map = {
                        'Mass': 'Msun',
                        'Teff': 'Teff',
                        'log(Teff)': 'logT',
                        'log(L/Lsun)': 'logL',
                        'log(R/Rsun)': 'logR',
                        'G-RP': 'G-Rp',
                        'G-V': 'G-V',
                        'G-J': 'G-J',
                        'BP-RP': 'Bp-Rp',
                        'J-H': 'J-H',
                        'H-K': 'H-Ks',
                        'G-K': 'G-Ks',
                        'V_mag': 'Mv'
                    }
    
                    SPOTS_expanded[f][age_SPOTS]['BCv'] = np.nan
                    start_index = len(self.SPOTS[f][age_SPOTS])
    
                    for bc_col, spots_col in columns_map.items():
                        SPOTS_expanded[f][age_SPOTS].loc[start_index:, bc_col] = MS_color_filtered[spots_col].values
    
                    SPOTS_expanded[f][age_SPOTS]['Lsun'] = 10**SPOTS_expanded[f][age_SPOTS]['log(L/Lsun)']
                    SPOTS_expanded[f][age_SPOTS].loc[start_index:, 'BCv'] = MS_color_filtered['BCv'].values
    
                    for i in range(self.SPOTS[f][age_SPOTS]['logAge'].index[0]+1, len(SPOTS_expanded[f][age_SPOTS]['logAge'])):
    
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'logAge'] = np.log10(age_SPOTS*1e9)
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'Fspot'] = float(f)/100
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'Age_Gyr'] = SPOTS_expanded[f][age_SPOTS]['Age_Gyr'][0]
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'Xspot'] = 0.8
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'log(T_hot)'] = SPOTS_expanded[f][age_SPOTS]['log(Teff)'][i]
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'log(T_cool)'] = 0.8*SPOTS_expanded[f][age_SPOTS]['log(Teff)'][i]
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'M_bol'] = M_bol_sun - 2.5 * SPOTS_expanded[f][age_SPOTS].loc[i, 'log(R/Rsun)']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs'] = SPOTS_expanded[f][age_SPOTS].loc[i, 'G-V'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'V_mag']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'J_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'G-J'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'RP_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'G-RP'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'BP_abs'] = SPOTS_expanded[f][age_SPOTS].loc[i, 'BP-RP'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'RP_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'H_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'J-H'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'J_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'K_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'G-K'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs']
    
    
                    for i, row in BTSettl_Li_isochrones_Teff.iterrows():
                        teff_value = row['Teff']
                        if i == 1 and BTSettl_Li_isochrones_Teff.at[0, 'A(Li)'] == -np.inf and row['A(Li)'] == -np.inf:
                            a_li_value = BTSettl_Li_isochrones_Teff.at[i + 1, 'A(Li)'] if i + 1 < len(BTSettl_Li_isochrones_Teff) else 0
                        else:
                            a_li_value = row['A(Li)']
                        nearest_index = self.find_nearest_index(SPOTS_expanded[f][age_SPOTS]['Teff'], teff_value)
                
                        
                        SPOTS_expanded[f][age_SPOTS].at[nearest_index, 'A(Li)'] = a_li_value
    
                    SPOTS_expanded[f][age_SPOTS]['RP-J'] = SPOTS_expanded[f][age_SPOTS]['RP_abs'] - SPOTS_expanded[f][age_SPOTS]['J_abs'] 
                    SPOTS_expanded[f][age_SPOTS]['G-H'] = SPOTS_expanded[f][age_SPOTS]['G_abs'] - SPOTS_expanded[f][age_SPOTS]['H_abs'] 
                    SPOTS_expanded[f][age_SPOTS] = SPOTS_expanded[f][age_SPOTS][~pd.isna(SPOTS_expanded[f][age_SPOTS]['A(Li)'])]
    
                else:
                    count = count + 1
    
                    age_index += 1
    
                    BTSettl_Li_isochrones_Teff = self.BTSettl[age_BTSettl][self.BTSettl[age_BTSettl]['Teff'] < Teff_SPOTS_min]
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff[BTSettl_Li_isochrones_Teff['Teff'] > min(BTSettl_Li_isochrones_Teff['Teff'])]
                    BTSettl_Li_isochrones_Teff = BTSettl_Li_isochrones_Teff.sort_values(by='Teff', ascending=False)
                    BTSettl_Li_isochrones_Teff['log(Teff)'] = np.log10(BTSettl_Li_isochrones_Teff['Teff'])
                    BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl] = BTSettl_Li_isochrones_Teff.reset_index()
    
                    MS_color_rows = {}
                    MS_color_rows[age_SPOTS] = pd.DataFrame(np.nan, index=range(n), columns=self.SPOTS[f][age_SPOTS].columns)
    
                    self.SPOTS[f][age_SPOTS]['G-V'] = self.SPOTS[f][age_SPOTS]['G_abs'] - self.SPOTS[f][age_SPOTS]['V_mag']
                    self.SPOTS[f][age_SPOTS]['log(Teff)'] = np.log10(self.SPOTS[f][age_SPOTS]['Teff'])
                    self.SPOTS[f][age_SPOTS]['M_bol'] = - 10 * np.log10(self.SPOTS[f][age_SPOTS]['Teff']/Teff_sun) - 5 * self.SPOTS[f][age_SPOTS]['log(R/Rsun)'] + M_bol_sun
                    self.SPOTS[f][age_SPOTS] = self.SPOTS[f][age_SPOTS].sort_values(by='log(Teff)', ascending=False)
    
                    SPOTS_expanded[f][age_SPOTS] = pd.concat([self.SPOTS[f][age_SPOTS], MS_color_rows[age_SPOTS]], ignore_index=True)
    
                    columns_map = {
                        'Mass': 'Msun',
                        'Teff': 'Teff',
                        'log(Teff)': 'logT',
                        'log(L/Lsun)': 'logL',
                        'log(R/Rsun)': 'logR',
                        'G-RP': 'G-Rp',
                        'G-V': 'G-V',
                        'G-J': 'G-J',
                        'BP-RP': 'Bp-Rp',
                        'J-H': 'J-H',
                        'H-K': 'H-Ks',
                        'G-K': 'G-Ks',
                        'V_mag': 'Mv'
                    }
    
                    SPOTS_expanded[f][age_SPOTS]['BCv'] = np.nan
                    start_index = len(self.SPOTS[f][age_SPOTS])
    
                    for bc_col, spots_col in columns_map.items():
                        SPOTS_expanded[f][age_SPOTS].loc[start_index:, bc_col] = MS_color_filtered[spots_col].values
    
                    SPOTS_expanded[f][age_SPOTS]['Lsun'] = 10**SPOTS_expanded[f][age_SPOTS]['log(L/Lsun)']
                    SPOTS_expanded[f][age_SPOTS].loc[start_index:, 'BCv'] = MS_color_filtered['BCv'].values
    
                    for i in range(self.SPOTS[f][age_SPOTS]['logAge'].index[0]+1, len(SPOTS_expanded[f][age_SPOTS]['logAge'])):
    
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'logAge'] = np.log10(age_SPOTS*1e9)
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'Fspot'] = float(f)/100
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'Age_Gyr'] = SPOTS_expanded[f][age_SPOTS]['Age_Gyr'][0]
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'Xspot'] = 0.8
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'log(T_hot)'] = SPOTS_expanded[f][age_SPOTS]['log(Teff)'][i]
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'log(T_cool)'] = 0.8*SPOTS_expanded[f][age_SPOTS]['log(Teff)'][i]
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'M_bol'] = M_bol_sun - 2.5 * SPOTS_expanded[f][age_SPOTS].loc[i, 'log(R/Rsun)']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs'] = SPOTS_expanded[f][age_SPOTS].loc[i, 'G-V'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'V_mag']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'J_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'G-J'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'RP_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'G-RP'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'BP_abs'] = SPOTS_expanded[f][age_SPOTS].loc[i, 'BP-RP'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'RP_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'H_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'J-H'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'J_abs']
                        SPOTS_expanded[f][age_SPOTS].loc[i, 'K_abs'] = - SPOTS_expanded[f][age_SPOTS].loc[i, 'G-K'] + SPOTS_expanded[f][age_SPOTS].loc[i, 'G_abs']
    
                    for i, row in BTSettl_Li_isochrones_Teff.iterrows():
                        teff_value = row['Teff']
                        if i == 1 and BTSettl_Li_isochrones_Teff.at[0, 'A(Li)'] == -np.inf and row['A(Li)'] == -np.inf:
                            a_li_value = BTSettl_Li_isochrones_Teff.at[i + 1, 'A(Li)'] if i + 1 < len(BTSettl_Li_isochrones_Teff) else 0
                        else:
                            a_li_value = row['A(Li)']
                        nearest_index = self.find_nearest_index(SPOTS_expanded[f][age_SPOTS]['Teff'], teff_value)
                        
                        SPOTS_expanded[f][age_SPOTS].at[nearest_index, 'A(Li)'] = a_li_value
                    
                    SPOTS_expanded[f][age_SPOTS]['RP-J'] = SPOTS_expanded[f][age_SPOTS]['RP_abs'] - SPOTS_expanded[f][age_SPOTS]['J_abs'] 
                    SPOTS_expanded[f][age_SPOTS]['G-H'] = SPOTS_expanded[f][age_SPOTS]['G_abs'] - SPOTS_expanded[f][age_SPOTS]['H_abs'] 
    
                    SPOTS_expanded[f][age_SPOTS] = SPOTS_expanded[f][age_SPOTS][~pd.isna(SPOTS_expanded[f][age_SPOTS]['A(Li)'])]
    
                SPOTS_expanded[f][age_SPOTS]['log(g)'] = self.calculate_log_g(SPOTS_expanded[f][age_SPOTS]['Mass'], 10**SPOTS_expanded[f][age_SPOTS]['log(R/Rsun)'])
        
        for f in SPOTS_expanded:
            if f != '00':
                for age_SPOTS in SPOTS_expanded[f]:
                    a_li = SPOTS_expanded[f][age_SPOTS]['A(Li)']
                    found_inf = False
                    for i in range(len(a_li) - 1):
                        if a_li.iloc[i] == -np.inf:
                            found_inf = True
                        if found_inf and a_li.iloc[i] != -np.inf and a_li.iloc[i] > a_li.iloc[i + 1]:
                            a_li.iloc[i] = -np.inf
        
        return SPOTS_expanded, BTSettl_Li_isochrones_Teff_dic
    
    def plot_CMD_ALi(self, SPOTS_expanded, BTSettl_Li_isochrones_Teff_dic, BTSettl_Li_isochrones, data_obs):
        cmap_BTSettl_original = plt.get_cmap('Reds')
        cmap_BTSettl_inverted = self.invert_cmap(cmap_BTSettl_original)
        
        cmap_SPOTS_original = plt.get_cmap('Blues')
        cmap_SPOTS_inverted = self.invert_cmap(cmap_SPOTS_original)
        
        norm_BTSettl = plt.Normalize(vmin=np.log10(0.004*1e9), vmax=np.log10(0.6*1e9))
        cmap_BTSettl = cm.ScalarMappable(norm=norm_BTSettl, cmap=cmap_BTSettl_inverted)
        
        norm_SPOTS = plt.Normalize(vmin=np.log10(0.004*1e9), vmax=np.log10(0.6*1e9))
        cmap_SPOTS = cm.ScalarMappable(norm=norm_SPOTS, cmap=cmap_SPOTS_inverted)
        
        for f in self.f_values:
        
            plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
            plt.rcParams['font.family'] = 'serif'
        
            fig, axs = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        
            bands1 = ['RP_abs', 'G_abs', 'G_abs', 'G_abs', 'G_abs', 'J_abs', 'H_abs', 'J_abs', 'H_abs']
            bands2 = ['J_abs', 'J_abs', 'H_abs', 'K_abs', 'RP_abs', 'H_abs', 'K_abs', 'H_abs', 'K_abs']
            colors = ['RP-J', 'G-J', 'G-H', 'G-K', 'G-RP', 'J-H', 'H-K', 'J-H', 'H-K']
        
            for i, ax in enumerate(axs.flat):
                band1 = bands1[i]
                band2 = bands2[i]
                color = colors[i]
                ax.scatter(data_obs[band1]-data_obs[band2], data_obs['ALi'], s=10, zorder=6, color='orange', alpha=0.5)
                ax.set_ylabel('$A(Li) [dex]$')
                ax.set_xlabel(f'{color} [mag]')
        
                max_array = []
        
                for age_BTSettl in BTSettl_Li_isochrones_Teff_dic[f].keys():
                    if age_BTSettl < 0.6:
                        log_age_BTSettl = np.log10(age_BTSettl * 1e9)
                        color_BTSettl = cmap_BTSettl.to_rgba(log_age_BTSettl)
                        plot_label = 'Low mass BT-Settl' if i == 0 and age_BTSettl == 0.004 else ""
                        ax.plot(BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][band1] - BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][band2], BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl]['A(Li)'], c=color_BTSettl, lw=1, alpha=1, label=plot_label)
                        max_array.append(max(BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][band1] - BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][band2]))
                        
                for age_SPOTS in SPOTS_expanded[f].keys():
                    if age_SPOTS < 0.6:
                        log_age_SPOTS = np.log10(age_SPOTS * 1e9)
                        color_SPOTS = cmap_SPOTS.to_rgba(log_age_SPOTS)
                        plot_label = f'SPOTS extended $f=0.${f}' if i == 0 and age_SPOTS == 0.004 else ""
                        ax.plot(SPOTS_expanded[f][age_SPOTS][color], SPOTS_expanded[f][age_SPOTS]['A(Li)'], lw=1, color=color_SPOTS, zorder=1, alpha=1, label=plot_label)
        
                ax.plot(BTSettl_Li_isochrones[0.120][band1] - BTSettl_Li_isochrones[0.120][band2], BTSettl_Li_isochrones[0.120]['A(Li)'], lw=1, c='k', ls='--',  alpha=1, label='Low mass BT-Settl 120 Myr' if i == 0 else "")
                ax.plot(SPOTS_expanded[f][0.126][color], SPOTS_expanded[f][0.126]['A(Li)'], lw=1, color='k', ls='-', zorder=2, alpha=1, label='SPOTS extended 120 Myr' if i == 0 else "")
        
                ax.set_xlim(0, max(max_array))
        
            fig.colorbar(cmap_BTSettl, ax=axs[:, 2], orientation='vertical', label=r'$\log({\rm Age\ [yr]})$')
            cbar = fig.colorbar(cmap_SPOTS, ax=axs[:, 2], orientation='vertical', label=r'$\log({\rm Age\ [yr]})$')
        
            cbar.ax.set_yticklabels([])
            cbar.ax.set_ylabel("")
        
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
        
            plt.show()
        
            plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
            plt.rcParams['font.family'] = 'serif'
        
            plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
            plt.rcParams['font.family'] = 'serif'
        
            fig, axs = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        
            bands1 = ['RP_abs', 'G_abs', 'G_abs', 'G_abs', 'G_abs', 'J_abs', 'H_abs', 'J_abs', 'H_abs']
            bands2 = ['J_abs', 'J_abs', 'H_abs', 'K_abs', 'RP_abs', 'H_abs', 'K_abs', 'H_abs', 'K_abs']
            bandsobs = ['RP_abs', 'G_abs', 'G_abs', 'G_abs', 'G_abs', 'J_abs', 'H_abs']
            colors = ['RP-J', 'G-J', 'G-H', 'G-K', 'G-RP', 'J-H', 'H-K', 'J-H', 'H-K']
        
        
            for i, ax in enumerate(axs.flat):
                band1 = bands1[i]
                band2 = bands2[i]
                bandobs = bandsobs[i]
                color = colors[i]
                ax.scatter(data_obs[band1]-data_obs[band2], data_obs[bandobs], s=10, zorder=6, color='orange', alpha=0.125)
                m = bandobs.split("_")[0]
                ax.set_ylabel(f'{m} [mag]')
                ax.set_xlabel(f'{color} [mag]')
        
                # Plot BTSettl and SPOTS isochrones
                for age_BTSettl in BTSettl_Li_isochrones_Teff_dic[f].keys():
                    log_age_BTSettl = np.log10(age_BTSettl * 1e9)
                    color_BTSettl = cmap_BTSettl.to_rgba(log_age_BTSettl)
                    plot_label = 'Low mass BT-Settl' if i == 0 and age_BTSettl == 0.004 else ""
                    ax.plot(BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][band1] - BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][band2], BTSettl_Li_isochrones_Teff_dic[f][age_BTSettl][bandobs], c=color_BTSettl, lw=1, alpha=1, label=plot_label)
        
                for age_SPOTS in SPOTS_expanded[f].keys():
                    log_age_SPOTS = np.log10(age_SPOTS * 1e9)
                    color_SPOTS = cmap_SPOTS.to_rgba(log_age_SPOTS)
                    plot_label = f'SPOTS extended $f=0.${f}' if i == 0 and age_SPOTS == 0.004 else ""
                    ax.plot(SPOTS_expanded[f][age_SPOTS][color], SPOTS_expanded[f][age_SPOTS][bandobs], lw=1, color=color_SPOTS, zorder=1, alpha=1, label=plot_label)
    
                ax.invert_yaxis()
    
                ax.plot(BTSettl_Li_isochrones[0.120][band1] - BTSettl_Li_isochrones[0.120][band2], BTSettl_Li_isochrones[0.120][bandobs], lw=1, c='k', ls='--', zorder=8, alpha=1, label='Low mass BT-Settl 120 Myr' if i == 0 else "")
                ax.plot(SPOTS_expanded[f][0.126][color], SPOTS_expanded[f][0.126][bandobs], lw=1, color='k', ls='-', zorder=7, alpha=1, label='SPOTS extended 120 Myr' if i == 0 else "")
    
            fig.colorbar(cmap_BTSettl, ax=axs[:, 2], orientation='vertical', label=r'$\log({\rm Age\ [yr]})$')
            cbar = fig.colorbar(cmap_SPOTS, ax=axs[:, 2], orientation='vertical', label=r'$\log({\rm Age\ [yr]})$')
    
            cbar.ax.set_yticklabels([])
            cbar.ax.set_ylabel("")
    
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))
    
            plt.show()
    
            plt.rcParams.update({'font.size': 11, 'axes.linewidth': 1, 'axes.edgecolor': 'k'})
            plt.rcParams['font.family'] = 'serif'

class Isochrones:
    def __init__(self, model1_dict, model2_dict, bands1, bands2, mod1, mod2, isochrones, filename=None, dpi=350, data_obs=None, obs=False, bandsobs=None):
        self.model1_dict = model1_dict
        self.model2_dict = model2_dict
        self.bands1 = bands1
        self.bands2 = bands2
        self.mod1 = mod1
        self.mod2 = mod2
        self.filename = filename
        self.dpi = dpi
        self.data_obs = data_obs
        self.obs = obs
        self.isochrones = isochrones
        self.bandsobs = bandsobs

    def get_first_word(self, string):
        return string.split()[0]

    def save_model_comparison(self, filename, mod1, mod2):
        mod1_first_word = self.get_first_word(mod1)
        mod2_first_word = self.get_first_word(mod2)
        final_filename = f"{filename}_{mod1_first_word}_{mod2_first_word}.pdf"
        plt.savefig(final_filename, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved figure as {final_filename}")

    def find_intersection(self, line1, line2):
        intersection_points = []
        for i in range(len(line1) - 1):
            for j in range(len(line2) - 1):
                path1 = Path([line1[i], line1[i+1]])
                path2 = Path([line2[j], line2[j+1]])
                if path1.intersects_path(path2):
                    x_int, y_int = self.get_intersection(line1[i], line1[i+1], line2[j], line2[j+1])
                    intersection_points.append((x_int, y_int))
        return intersection_points

    def get_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None, None
        x_int = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        y_int = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return x_int, y_int

    def plot_isochrones_grid(self):
        plt.rcParams.update({'font.size': 14})  # Set the font size
    
        fig, axs = plt.subplots(len(self.isochrones), len(self.bands1), figsize=(8*len(self.bands1), 6*len(self.isochrones)), sharex='col')  # Create the figure and subplots
    
        max_mass_labels = 5  # Maximum number of mass labels to show
        handles, labels = [], []
    
        n = 0
    
        # Iterate over each column (band combination)
        for i, (band1, band2) in enumerate(zip(self.bands1, self.bands2)):
            max_x_column = float('-inf')
            max_y = float('-inf')
    
            # Iterate over each row (isochrone)
            for j, isochrone in enumerate(self.isochrones):
                ax = axs[j, i]
                isochrone_1 = min(self.model1_dict.keys(), key=lambda x: abs(x - isochrone))
                isochrone_2 = min(self.model2_dict.keys(), key=lambda x: abs(x - isochrone))
    
                model1_df = self.model1_dict[isochrone_1]
                model1_df = model1_df.loc[model1_df['M/Ms'] < 1.5]
                model1_df = model1_df.loc[model1_df['M/Ms'] > 0.08]
    
                model2_df = self.model2_dict[isochrone_2]
                model2_df = model2_df.loc[model2_df['M/Ms'] < 1.5]
                model2_df = model2_df.loc[model2_df['M/Ms'] > 0.08]
    
                min_mm_model1 = min(model1_df['M/Ms'])
                min_mm_model2 = min(model2_df['M/Ms'])
    
                mm_values = np.linspace(1.5, min(min_mm_model1, min_mm_model2), max_mass_labels)
    
                x1 = self.model1_dict[isochrone_1][band1[0]] - self.model1_dict[isochrone_1][band1[1]]
                y1 = self.model1_dict[isochrone_1][band1[0]]
                x2 = self.model2_dict[isochrone_2][band2[0]] - self.model2_dict[isochrone_2][band2[1]]
                y2 = self.model2_dict[isochrone_2][band2[0]]
    
                line1, = ax.plot(x1, y1, label=f'{self.get_first_word(self.mod1)}', zorder=2, linewidth=2, color='blue')
                line2, = ax.plot(x2, y2, label=f'{self.get_first_word(self.mod2)}', zorder=1, linewidth=2, color='orange')
    
                if i == 0 and j == 0:  # Add handles and labels only once
                    handles.extend([line1, line2])
                    labels.extend([line1.get_label(), line2.get_label()])
    
                max_x_column = max(max_x_column, np.max(x1), np.max(x2))
                max_y = max(max_y, np.max(y1), np.max(y2))
    
                for mm_value in mm_values:
                    closest_row_model1 = model1_df.iloc[(model1_df['M/Ms'] - mm_value).abs().argsort()[:1]]
                    closest_row_model2 = model2_df.iloc[(model2_df['M/Ms'] - mm_value).abs().argsort()[:1]]
    
                    x_mass_model1 = closest_row_model1[band1[0]] - closest_row_model1[band1[1]]
                    y_mass_model1 = closest_row_model1[band1[0]]
                    mass_model1 = closest_row_model1['M/Ms']
    
                    x_mass_model2 = closest_row_model2[band2[0]] - closest_row_model2[band2[1]]
                    y_mass_model2 = closest_row_model2[band2[0]]
                    mass_model2 = closest_row_model2['M/Ms']
    
                    for x_mass1, y_mass1, mass1, x_mass2, y_mass2, mass2 in zip(x_mass_model1, y_mass_model1, mass_model1, x_mass_model2, y_mass_model2, mass_model2):
                        text = f'{((mass1+mass2)/2):.3f}'
    
                        if x_mass2 - x_mass1 < 0:
                            text_x = x_mass1 + (1/30)*(max_x_column+0.2)
                            ax.text(text_x, y_mass1, text, fontsize=12, color='k')
                        elif x_mass1 - x_mass2 < 0:
                            text_x = x_mass2 + (1/30)*(max_x_column+0.2)
                            ax.text(text_x, y_mass2, text, fontsize=12, color='k')
    
                        ax.plot([x_mass1, x_mass2], [y_mass1, y_mass2], color='k', linewidth=1)
    
                intersection_points = self.find_intersection(np.column_stack((x1, y1)), np.column_stack((x2, y2)))
    
                if intersection_points:
                    first_point = intersection_points[0]
                    if -0.2 <= first_point[0] <= max_x_column and 1 <= first_point[1] <= max_y:
                        ax.plot([-0.2, first_point[0]], [first_point[1], first_point[1]], color='black', linewidth=1, linestyle='--')
                        ax.plot([first_point[0], first_point[0]], [max_y, first_point[1]], color='black', linewidth=1, linestyle='--')
                        ax.text(0.55, 0.95, f'({first_point[0]:.2f}, {first_point[1]:.2f})', fontsize=18, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='black')  # Adjusted position for text
                    else:
                        pass
    
                # Plot observational data if available
                if self.obs and self.bandsobs:
                    if isochrone == 0.08 or isochrone == 0.12:
                        obs_points = ax.errorbar(self.data_obs[f'{self.bandsobs[i][0]}'+'_abs']-self.data_obs[f'{self.bandsobs[i][1]}'+'_abs'],
                                    self.data_obs[f'{self.bandsobs[i][0]}'+'_abs'], 
                                    yerr=self.data_obs['e_'+f'{self.bandsobs[i][0]}'], fmt='.', alpha=0.25, label='Obs', zorder=0, color='r')
                        if n == 0:  # Add observational points handle and label only once
                            handles.append(obs_points)
                            labels.append('Obs')
                            n = n + 1
    
                if j == len(self.isochrones) - 1:
                    ax.set_xlabel(f'{band1[0].split("_")[0]}-{band1[1].split("_")[0]} [mag]')
                if i == 0:
                    ax.set_ylabel(f'{band1[0].split("_")[0]} [mag]')
                if j == 0:
                    ax.set_title(f'{band1[0].split("_")[0]}-{band1[1].split("_")[0]}')
    
                ax.text(0.95, 0.95, f'{isochrone*1000:.0f} Myr', transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
                ax.set_xlim(-0.2, max_x_column+0.2)
                ax.set_ylim(max_y, 1)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
        # Create a single legend for the entire figure
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.075), fontsize=12, ncol=3)
    
        plt.subplots_adjust(wspace=0.125, hspace=0)  # Adjust top to make space for the legend
    
        if self.filename:
            self.save_model_comparison(self.filename, self.mod1, self.mod2)
        else:
            plt.show()
        
        plt.rcParams.update({'font.size': 14})  # Set the font size
    
    def plot_single_column(self, band1, band2, bandobs):
        plt.rcParams.update({'font.size': 14})  
        
        fig, axs = plt.subplots(len(self.isochrones), 1, figsize=(10, 6*len(self.isochrones)), sharex='col')  

        max_mass_labels = 5  

        for j, isochrone in enumerate(self.isochrones):
            ax = axs[j]
            isochrone_1 = min(self.model1_dict.keys(), key=lambda x: abs(x - isochrone))
            isochrone_2 = min(self.model2_dict.keys(), key=lambda x: abs(x - isochrone))

            model1_df = self.model1_dict[isochrone_1]
            model1_df = model1_df.loc[model1_df['M/Ms'] < 1.5]
            model1_df = model1_df.loc[model1_df['M/Ms'] > 0.08]

            model2_df = self.model2_dict[isochrone_2]
            model2_df = model2_df.loc[model2_df['M/Ms'] < 1.5]
            model2_df = model2_df.loc[model2_df['M/Ms'] > 0.08]

            min_mm_model1 = min(model1_df['M/Ms'])
            min_mm_model2 = min(model2_df['M/Ms'])

            mm_values = np.linspace(1.5, min(min_mm_model1, min_mm_model2), max_mass_labels)

            x1 = self.model1_dict[isochrone_1][band1[0]] - self.model1_dict[isochrone_1][band1[1]]
            y1 = self.model1_dict[isochrone_1][band1[2]]
            x2 = self.model2_dict[isochrone_2][band2[0]] - self.model2_dict[isochrone_2][band2[1]]
            y2 = self.model2_dict[isochrone_2][band2[2]]

            ax.plot(x1, y1, label=f'{self.get_first_word(self.mod1)}', zorder=2, linewidth=2, color='blue')
            ax.plot(x2, y2, label=f'{self.get_first_word(self.mod2)}', zorder=1, linewidth=2, color='orange')

            max_x_column = max(np.max(x1), np.max(x2))
            max_y = max(np.max(y1), np.max(y2))

            for mm_value in mm_values:
                closest_row_model1 = model1_df.iloc[(model1_df['M/Ms'] - mm_value).abs().argsort()[:1]]
                closest_row_model2 = model2_df.iloc[(model2_df['M/Ms'] - mm_value).abs().argsort()[:1]]

                x_mass_model1 = closest_row_model1[band1[0]] - closest_row_model1[band1[1]]
                y_mass_model1 = closest_row_model1[band1[2]]
                mass_model1 = closest_row_model1['M/Ms']

                x_mass_model2 = closest_row_model2[band2[0]] - closest_row_model2[band2[1]]
                y_mass_model2 = closest_row_model2[band2[2]]
                mass_model2 = closest_row_model2['M/Ms']

                for x_mass1, y_mass1, mass1, x_mass2, y_mass2, mass2 in zip(x_mass_model1, y_mass_model1, mass_model1, x_mass_model2, y_mass_model2, mass_model2):
                    text = f'{((mass1+mass2)/2):.3f}'

                    if x_mass2 - x_mass1 < 0:
                        text_x = x_mass1 + (1/30)*(max_x_column+0.2)
                        ax.text(text_x, y_mass1, text, fontsize=12, color='k')
                    elif x_mass1 - x_mass2 < 0:
                        text_x = x_mass2 + (1/30)*(max_x_column+0.2)
                        ax.text(text_x, y_mass2, text, fontsize=12, color='k')

                    ax.plot([x_mass1, x_mass2], [y_mass1, y_mass2], color='k', linewidth=1)
            intersection_points = self.find_intersection(np.column_stack((x1, y1)), np.column_stack((x2, y2)))

            if intersection_points:
                first_point = intersection_points[0]
                if -0.2 <= first_point[0] <= max_x_column and 1 <= first_point[1] <= max_y:
                    ax.plot([-0.2, first_point[0]], [first_point[1], first_point[1]], color='black', linewidth=1, linestyle='--')
                    ax.plot([first_point[0], first_point[0]], [max_y, first_point[1]], color='black', linewidth=1, linestyle='--')
                    ax.text(0.55, 0.95, f'({first_point[0]:.2f}, {first_point[1]:.2f})', fontsize=18, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='black') 
                else:
                    pass

            if self.obs and (isochrone == 0.08 or isochrone == 0.12):  
                ax.errorbar(self.data_obs[f'{bandobs[0]}'+'_abs']-self.data_obs[f'{bandobs[1]}'+'_abs'],
                            self.data_obs[f'{bandobs[2]}'+'_abs'], 
                            yerr=self.data_obs['e_'+f'{bandobs[2]}'], fmt='.', alpha=0.125, label='Obs', zorder=0, color='r')

            ax.set_xlabel(f'{band1[0].split("_")[0]}-{band1[1].split("_")[0]} [mag]')
            ax.set_ylabel(f'{band1[2].split("_")[0]} [mag]')
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.text(0.825, 0.85, f'{isochrone_1} Gyr', transform=ax.transAxes, fontsize=18, verticalalignment='bottom', horizontalalignment='left')  

            ax.set_xlim(-0.2, 6)
            ax.set_ylim(1, max_y)
            ax.invert_yaxis()

        plt.subplots_adjust(hspace=0)
        fig.suptitle(f'CMD; {self.mod1} vs {self.mod2}', y=0.9075)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        legend_handles = [handles[labels.index(label)] for label in unique_labels]
        if self.obs:
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Obs'))  
        fig.legend(legend_handles, unique_labels + ['Obs'], loc='upper center', bbox_to_anchor=(0.5, 0.075), fontsize=16, ncol=3, edgecolor='black')  
        if self.filename:
            self.save_model_comparison(self.filename, self.mod1, self.mod2)
        else:
            plt.show()
            
        plt.rcParams.update({'font.size': 14})
    
    def plot_single_column_Teff(self):
        plt.rcParams.update({'font.size': 14})  

        fig, axs = plt.subplots(len(self.isochrones), 1, figsize=(10, 6*len(self.isochrones)), sharex='col')  

        max_mass_labels = 5

        for j, isochrone in enumerate(self.isochrones):
            ax = axs[j]
            isochrone_1 = min(self.model1_dict.keys(), key=lambda x: abs(x - isochrone))
            isochrone_2 = min(self.model2_dict.keys(), key=lambda x: abs(x - isochrone))

            model1_df = self.model1_dict[isochrone_1]
            model1_df = model1_df.loc[model1_df['M/Ms'] < 1.5]
            model1_df = model1_df.loc[model1_df['M/Ms'] > 0.08]

            model2_df = self.model2_dict[isochrone_2]
            model2_df = model2_df.loc[model2_df['M/Ms'] < 1.5]
            model2_df = model2_df.loc[model2_df['M/Ms'] > 0.08]

            min_mm_model1 = min(model1_df['M/Ms'])
            min_mm_model2 = min(model2_df['M/Ms'])

            mm_values = np.linspace(1.5, min(min_mm_model1, min_mm_model2), max_mass_labels)

            x1 = self.model1_dict[isochrone_1]['Teff']
            y1 = np.log10(self.model1_dict[isochrone_1]['Lsun'])
            x2 = self.model2_dict[isochrone_2]['Teff']
            y2 = np.log10(self.model2_dict[isochrone_2]['Lsun'])

            ax.plot(x1, y1, label=f'{self.get_first_word(self.mod1)}', zorder=2, linewidth=2, color='blue')
            ax.plot(x2, y2, label=f'{self.get_first_word(self.mod2)}', zorder=1, linewidth=2, color='orange')

            for mm_value in mm_values:
                closest_row_model1 = model1_df.iloc[(model1_df['M/Ms'] - mm_value).abs().argsort()[:1]]
                closest_row_model2 = model2_df.iloc[(model2_df['M/Ms'] - mm_value).abs().argsort()[:1]]

                x_mass_model1 = closest_row_model1['Teff']
                y_mass_model1 = np.log10(closest_row_model1['Lsun'])
                mass_model1 = closest_row_model1['M/Ms']

                x_mass_model2 = closest_row_model2['Teff']
                y_mass_model2 = np.log10(closest_row_model2['Lsun'])
                mass_model2 = closest_row_model2['M/Ms']

                for x_mass1, y_mass1, mass1, x_mass2, y_mass2, mass2 in zip(x_mass_model1, y_mass_model1, mass_model1, x_mass_model2, y_mass_model2, mass_model2):
                    text = f'{((mass1+mass2)/2):.3f}'

                    if y_mass2+10 - (y_mass1+10) < 0:
                        text_x = x_mass1 + (1/10)*(8000-1000)
                        ax.text(text_x, y_mass1-0.25, text, fontsize=12, color='k')
                    elif y_mass2+10 - (y_mass1+10) > 0:
                        text_x = x_mass2 + 75
                        ax.text(text_x, y_mass2+0.25, text, fontsize=12, color='k')

                    ax.plot([x_mass1, x_mass2], [y_mass1, y_mass2], color='k', linewidth=1)

            intersection_points = self.find_intersection(np.column_stack((x1, y1)), np.column_stack((x2, y2)))

            if intersection_points:
                first_point = intersection_points[0]
                if 1000 <= first_point[0] <= 8000 and -4.5 <= first_point[1] <= 1.2:
                    ax.plot([first_point[0], 8200], [first_point[1], first_point[1]], color='black', linewidth=1, linestyle='--')
                    ax.plot([first_point[0], first_point[0]], [-4.5, first_point[1]], color='black', linewidth=1, linestyle='--')
                    ax.text(0.55, 0.95, f'({first_point[0]:.2f}, {first_point[1]:.2f})', fontsize=18, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='black') 
                else:
                    pass

            if self.obs and (isochrone == 0.08 or isochrone == 0.12):  
                ax.scatter(self.data_obs['Teff_x'],
                            np.log10(self.data_obs['Lsun']),
                            alpha=0.125, label='Obs', s=10, zorder=0, color='r')

            ax.set_xlabel('$T_{eff}$ [K]')
            ax.set_ylabel('$\log{(\mathcal{L}/\mathcal{L}_{\odot})}$')
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.text(0.825, 0.85, f'{round(isochrone_1, 3)} Gyr', transform=ax.transAxes, fontsize=18, verticalalignment='bottom', horizontalalignment='left')  

            ax.set_xlim(2000, 8200)
            ax.set_ylim(-3.75, 1.35)
            ax.invert_xaxis()

        plt.subplots_adjust(hspace=0)
        fig.suptitle(f'HRD; {self.mod1} vs {self.mod2}', y=0.9075)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        legend_handles = [handles[labels.index(label)] for label in unique_labels]
        if self.obs:
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Obs'))  
        fig.legend(legend_handles, unique_labels + ['Obs'], loc='upper center', bbox_to_anchor=(0.5, 0.075), fontsize=16, ncol=3, edgecolor='black')
        if self.filename:
            self.save_model_comparison(self.filename, self.mod1, self.mod2)
        else:
            plt.show()
    
        plt.rcParams.update({'font.size': 14})  


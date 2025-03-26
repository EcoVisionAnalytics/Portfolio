# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:40:29 2024

@author: EcoVision Analytics
@project: Montana State University, Bird Ecological Health in the GYE
@PI: Andy Hansen, hansen@montana.edu
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import json


with open('config_plot.json', 'r') as config_file:
    config = json.load(config_file)

os.chdir(config['working_directory'])

abun_df = pd.read_csv(config['abundance_file'])
trend_df = pd.read_csv(config['trend_file'])

abun_80_df = pd.read_csv(config['abundance_80_file'])
abun_90_df = pd.read_csv(config['abundance_90_file'])

trend_80_df = pd.read_csv(config['trend_80_file'])
trend_90_df = pd.read_csv(config['trend_90_file'])

output_dir = config['output_directory']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

species_list = abun_df['species'].unique()

for species in species_list:
    species_abun_df = abun_df[(abun_df['species'] == species) & (abun_df['region'] == 'continent')]
    species_trend_df = trend_df[(trend_df['species'] == species) & (trend_df['region'] == 'continent')]

    species_abun_80_df = abun_80_df[(abun_80_df['species'] == species) & (abun_80_df['region'] == 'continent')]
    species_trend_80_df = trend_80_df[(trend_80_df['species'] == species) & (trend_80_df['region'] == 'continent')]

    species_abun_90_df = abun_90_df[(abun_90_df['species'] == species) & (abun_90_df['region'] == 'continent')]
    species_trend_90_df = trend_90_df[(trend_90_df['species'] == species) & (trend_90_df['region'] == 'continent')]

    if species_abun_df.empty or species_trend_df.empty:
        continue

    years = species_abun_df['year']
    index = species_abun_df['index']  
    index_q_0_05 = species_abun_df['index_q_0.05']
    index_q_0_95 = species_abun_df['index_q_0.95']
    obs_mean = species_abun_df['obs_mean']  

    mean_n_routes_70 = species_abun_df['n_routes'].mean()
    mean_n_routes_80 = species_abun_80_df['n_routes'].mean() if not species_abun_80_df.empty else "N/A"
    mean_n_routes_90 = species_abun_90_df['n_routes'].mean() if not species_abun_90_df.empty else "N/A"

    trend_value = species_trend_df['trend'].values[0]
    trend_q_0_05 = species_trend_df['trend_q_0.05'].values[0]
    trend_q_0_95 = species_trend_df['trend_q_0.95'].values[0]

    if trend_q_0_05 > 0 or trend_q_0_95 < 0:
        significance = "Significant at 90% CI"
    else:
        significance = "Not significant at 90% CI"

    if trend_q_0_05 > 0 or trend_q_0_95 < 0:
        significance_95 = "Significant at 95% CI"
    else:
        significance_95 = "Not significant at 95% CI"

    plt.figure(figsize=(14, 10))

    plt.plot(years, index, label="1970-2022 Index", color="#0072B2", linewidth=3)

    plt.plot(species_abun_80_df['year'], species_abun_80_df['index'], label="1980-2022 Index", color="#D55E00", linewidth=3)

    plt.plot(species_abun_90_df['year'], species_abun_90_df['index'], label="1990-2022 Index", color="#009E73", linewidth=3)

    plt.scatter(years, obs_mean, color="#CC79A7", label="Observed Mean", zorder=5)

    plt.xlabel("Year")
    plt.ylabel("Relative Abundance")
    plt.title(f"{species} in the GYE, 1970-2022", fontsize=16, fontweight='bold')

    textstr = f"Trend : {trend_value:.2f}\nMean Routes: {mean_n_routes_70:.1f}\n{significance}\n{significance_95}"

    plt.subplots_adjust(right=0.75) 

    cell_text = [
        [f"{trend_value:.2f}", f"{mean_n_routes_70:.1f}", "Significant" if trend_q_0_05 > 0 or trend_q_0_95 < 0 else "Not significant", "Significant" if trend_q_0_05 > 0 or trend_q_0_95 < 0 else "Not significant"],
        [f"{species_trend_80_df['trend'].values[0]:.2f}" if not species_trend_80_df.empty else "N/A",
         f"{mean_n_routes_80:.1f}" if mean_n_routes_80 != "N/A" else "N/A",
         "Significant" if not species_trend_80_df.empty and (species_trend_80_df['trend_q_0.05'].values[0] > 0 or species_trend_80_df['trend_q_0.95'].values[0] < 0) else "Not significant",
         "Significant" if not species_trend_80_df.empty and (species_trend_80_df['trend_q_0.05'].values[0] > 0 or species_trend_80_df['trend_q_0.95'].values[0] < 0) else "Not significant"],
        [f"{species_trend_90_df['trend'].values[0]:.2f}" if not species_trend_90_df.empty else "N/A",
         f"{mean_n_routes_90:.1f}" if mean_n_routes_90 != "N/A" else "N/A",
         "Significant" if not species_trend_90_df.empty and (species_trend_90_df['trend_q_0.05'].values[0] > 0 or species_trend_90_df['trend_q_0.95'].values[0] < 0) else "Not significant",
         "Significant" if not species_trend_90_df.empty and (species_trend_90_df['trend_q_0.05'].values[0] > 0 or species_trend_90_df['trend_q_0.95'].values[0] < 0) else "Not significant"]
    ]
    rows = ["1970-2022", "1980-2022", "1990-2022"]
    columns = ["Trend", "Mean Routes", "90% CI", "95% CI"]

    table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom', cellLoc='center', bbox=[0.0, -0.4, 1.0, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_text_props(fontweight='bold')

    plt.legend(loc='upper right', title="Legend")

    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.85)  

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{species}_abundance_plot.png"))

    plt.close()

print(f"Plots executed and saved in the directory: {output_dir}")
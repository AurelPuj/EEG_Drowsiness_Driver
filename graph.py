# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:07:56 2020

@author: Aurelien
"""

import matplotlib.pyplot as plt

def plot_band_graph():
    #On affiche chaque cannal sur un subplot
    fig,axs = plt.subplots(6,3)
    row = 0
    color=['red','darkblue','orange','purple','darkgreen','brown','lightgreen','lightblue','magenta','orangered','salmon','blue']
    border=[0.05,0.95,0.6,0.2]
    fig.subplots_adjust(border[0],border[0],border[1],border[1],border[-1],border[-2])
    plt.show()

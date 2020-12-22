# -*- coding: utf-8 -*-
"""

Copyright:
Wei-Long Zheng and Bao-Liang Lu
Center for Brain-like Computing and Machine Intelligence, Department of Computer Science and Engineering, Shanghai Jiao Tong University, China
Key Laboratory of Shanghai Education Commission for Intelligent Interaction and Cognitive Engineering, Shanghai Jiao Tong University, China
Brain Science and Technology Research Center, Shanghai Jiao Tong University, China

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

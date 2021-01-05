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

def plot_band_graph(df_raw_data,file_name):
    
    fig,axs = plt.subplots(6,3)
    row = 0
    color=['red','darkblue','orange','purple','darkgreen','brown','lightgreen','lightblue','magenta','orangered','salmon','blue']
    count = 0
    row = 0
    
    for band in df_raw_data.columns :
        if count%3 == 0 and count>0 :
            row+=1
        axs[row,count%3].plot(df_raw_data[band],color=color[count%12])
        axs[row,count%3].set_title(band,color=color[count%12])
        count += 1
    
    border=[0.05,0.95,0.6,0.2]
    fig.subplots_adjust(border[0],border[0],border[1],border[1],border[-1],border[-2])
    plt.show()
    
    fig.savefig("..\\DataBase\\SEED-VIG\\Figure_Raw_Data\\"+file_name+'.pdf')


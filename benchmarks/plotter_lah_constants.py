import sys

import matplotlib.pyplot as plt
from pandas import read_csv


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # For talks, use sans-serif
    "font.size": 26,
    # "font.size": 16,
})
cmap = plt.cm.Set1
colors = cmap.colors
titles_2_colors = dict(cold_start='black', 
                       nearest_neighbor=colors[6], 
                       silver=colors[2],
                       nesterov='black',
                       learned_no_accel=colors[0], #colors[3],
                       l2ws10000=colors[3],
                       learned_no_accel_safeguard=colors[3], #colors[0],
                       lm10000=colors[0],
                       lah_accel=colors[1],
                       lah_accel_1=colors[1],
                       lah_accel_2=colors[1],
                       lah_accel_3=colors[1],
                       conj_grad=colors[8],
                       prev_sol=colors[2],
                       reg_k0=colors[3],
                       reg_k5=colors[0],
                       reg_k30=colors[5],
                       reg_k60=colors[2],
                    #    reg_k120=colors[0],
                       obj_k0=colors[3],
                       obj_k5=colors[0],
                       obj_k15=colors[1],
                       obj_k30=colors[5],
                       obj_k60=colors[2])
titles_2_colors['UB97.5'] = colors[1]
titles_2_colors['UB87.5'] = colors[1]
titles_2_colors['LB2.5'] = colors[1]
                    #    obj_k120='gray')
titles_2_styles = dict(cold_start='-', 
                       nearest_neighbor='-',
                       nesterov='-',
                       silver='-', 
                       learned_no_accel='-',
                       l2ws10000='-',
                       learned_no_accel_safeguard='-',
                       lm10000='-',
                       lah_accel='-',
                       lah_accel_1='-',
                       lah_accel_2='-',
                       lah_accel_3='-',
                       conj_grad='-',
                       prev_sol='-',
                       reg_k0='-',
                       reg_k5='-',
                       reg_k30='-',
                       reg_k60='-',
                       reg_k120='-',
                       obj_k0='-',
                       obj_k5='-',
                       obj_k15='-',
                       obj_k30='-',
                       obj_k60='-')
titles_2_styles['UB97.5'] = ':'
titles_2_styles['UB87.5'] = ':'
titles_2_styles['LB2.5'] = ':'
                    #    obj_k120='-')
titles_2_markers = dict(cold_start='v', 
                       nearest_neighbor='<', 
                       nesterov='^',
                       silver='D',
                       learned_no_accel='o',
                       l2ws10000='>',
                       learned_no_accel_safeguard='o',
                       lm10000='o',
                       lah_accel='s',
                       lah_accel_1='s',
                       lah_accel_2='s',
                       lah_accel_3='s',
                       conj_grad='X',
                       prev_sol='^',
                       reg_k0='>',
                       reg_k5='o',
                       reg_k30='x',
                       reg_k60='D',
                    #    reg_k120='-',
                       obj_k0='>',
                       obj_k5='o',
                       obj_k15='s',
                       obj_k30='x',
                       obj_k60='D')
titles_2_markers['UB97.5'] = 'None'
titles_2_markers['UB87.5'] = 'None'
titles_2_markers['LB2.5'] = 'None'
titles_2_marker_starts = dict(cold_start=0, 
                       nearest_neighbor=16, 
                       silver=20,
                       nesterov=23,
                       learned_no_accel=8,
                       l2ws10000=8,
                       learned_no_accel_safeguard=4,
                       lm10000=4,
                       lah_accel=12,
                       lah_accel_1=12,
                       lah_accel_2=12,
                       lah_accel_3=12,
                       conj_grad=0,
                       prev_sol=23,
                       reg_k0=8,
                       reg_k5=4,
                       reg_k30=0,
                       reg_k60=20,
                    #    reg_k120='-',
                       obj_k0=8,
                       obj_k5=4,
                       obj_k15=12,
                       obj_k30=0,
                       obj_k60=20)

titles_2_marker_starts['UB97.5'] = 0
titles_2_marker_starts['UB87.5'] = 0
titles_2_marker_starts['LB2.5'] = 0
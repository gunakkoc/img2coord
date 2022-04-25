# -*- coding: utf-8 -*-
"""
Helmholtz-Institute Erlangen-Nürnberg for Renewable Energy (IEK-11)
Forschungszentrum Jülich

2022

@author: Gun Deniz Akkoc
"""

from img2coord import img2coord

#Initalize the class
m = img2coord("example.vk4") 

#Change settings
m.bw_threshold = 30
m.circ_threshold = 0.9
m.gauss_window_size = 7
m.min_spot_r = 0.4
m.max_spot_r = 1.

#Start spot detection
m.start()

#Export results
m.export_coordinates() #if no file path is given, it is assigned automatically
m.export_rgb_detected("detected_spots.png") #file path can also be provided
m.export_areas()
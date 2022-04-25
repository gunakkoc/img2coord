# -*- coding: utf-8 -*-
"""
Helmholtz-Institute Erlangen-Nürnberg for Renewable Energy (IEK-11)
Forschungszentrum Jülich

2022

@author: Gun Deniz Akkoc
"""

import vk4extract
import numpy as np
import cv2 as cv
from pathlib import Path
import os

class img2coord():
    #User options
    filepath=None #path to the vk4 file
    bw_threshold=20 #black-white threshold, ranges between 0 and 255
    gauss_window_size=5 #Window size for Gaussian smoothing, a window of n x n will be used
    min_spot_r=0.5 #minimum spot radius (mm)
    max_spot_r=1.2 #maximum spot radius (mm)
    circ_threshold=0.8 #circularity threshold, ranges between 0 and 1
    dist_btw_spots=3.4 #distance between spots, particularly between rows
    closing_enabled = False #toggle morphological closing
    closing_window_size = 3 #window size for morphological closing, ignored if morphological closing is not used
    cannysoft = 100 #soft threshold for canny edge detection, ignored if canny is not used
    cannyhard = 200 #hard threshold for canny edge detection, ignored if canny is not used

    #Initalizations
    bg = np.asarray([0,0,0]) #RGB values of background
    _line_width = 3
    _font_size = 12
    
    def __init__(self,filepath=None):
        if filepath:
            self.filepath = filepath

    def load_vk4(self,filepath = None): 
        #Load image from VK4
        self.fullpath = self.filepath
        self.targetdir,self.fname = os.path.split(self.fullpath)
        if not self.targetdir:
            self.targetdir = "."
        self.fname = os.path.splitext(self.fname)[0]
        
        #Read measurement conditions and image
        with open(self.fullpath, 'rb') as in_file:
            offsets = vk4extract.extract_offsets(in_file)
            rgb_dict = vk4extract.extract_color_data(offsets, 'light', in_file)
            conds = vk4extract.extract_measurement_conditions(offsets,in_file)
        
        #Extract RGB microscope image
        rgb_data = rgb_dict['data']
        self.height = rgb_dict['height']
        self.width = rgb_dict['width']
        self.raw = np.reshape(rgb_data, (self.height, self.width, 3))
        self.org = self.raw.copy()
        # cv.imwrite(self.targetdir + "/" + self.fname + '.png', self.rgb)
    
        #Get picometers per pixel
        self.orgpxtopm = conds['x_length_per_pixel']
        self.pxtopm = conds['x_length_per_pixel']
        self.pmtomm = 0.000000001
        self.mmtopm = 1000000000
        self.pxtomm = self.pxtopm * self.pmtomm
        self.mmtopx = 1 / self.pxtomm
        
        #Scale visual elements according to image size
        self._line_width = np.ceil(self.height * self.width * 0.00000025)
        self._font_size = np.ceil(self.height * self.width * 0.00000015)
        
        return self.org
    
    def rescale(self,ratio = 1.0): #Rescale the image
        new_w = int(self.raw.shape[1] * ratio)
        new_h = int(self.raw.shape[0] * ratio)
        self.org = cv.resize(self.raw,(new_w,new_h))
        self.width = new_w
        self.height = new_h
        
        #Adjust px to mm conversions accordingly
        self.pxtopm = self.orgpxtopm / (new_w / self.raw.shape[1])
        self.pxtomm = self.pxtopm * self.pmtomm
        self.mmtopx = 1 / self.pxtomm
        
        #Scale visual elements (again) accordingly
        self._line_width = np.ceil(self.height * self.width * 0.00000025)
        self._font_size = np.ceil(self.height * self.width * 0.00000015)

    def get_rect_mean(self,x1=None,y1=None,x2=None,y2=None):
        #Calculate average RGB value of a rectangle within the image
        if x1 is None:
            x1 = 0
            y1 = 0
            x2 = self.width * 0.05
            y2 = self.height * 0.05
        self.rgb = self.org.copy()
        self.rgb = cv.GaussianBlur(self.rgb,(self.gauss_window_size,self.gauss_window_size),0)
        bg = self.rgb[int(y1):int(y2),int(x1):int(x2)]
        bg = np.float64(bg)
        self.bg = np.mean(bg,(0,1))
        self.rgb = self.org.copy()
        return self.bg
    
    def bg_substraction(self): #Substract background from the image
        self.rgb = self.org.copy()
        #Gaussian smoothing
        self.rgb = cv.GaussianBlur(self.rgb,(self.gauss_window_size,self.gauss_window_size),0)
        #Calculate Euclidian distance of each pixel to background RGB
        rgb_64 = np.float64(self.rgb)
        rgb_64 = rgb_64 - self.bg
        gray = np.sqrt(np.sum(np.multiply(rgb_64,rgb_64),2))
        #Min-max scaling of resulting gray scale image between 0 and 255
        gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
        self.gray = np.uint8(gray * 255)
        return self.gray
        
    def bw_by_threshold(self):
        #Get black-white by applying a threshold on gray image
        retval,self.bw = cv.threshold(self.gray,self.bw_threshold,255,cv.THRESH_BINARY)
        return self.bw
        
    def morph_closing(self):
        #Apply morphological closing on a BW image
        se = np.ones((self.closing_window_size,self.closing_window_size), dtype='uint8')
        self.bw = cv.morphologyEx(self.bw, cv.MORPH_CLOSE, se)
        return self.bw
    
    def bw_by_canny(self,gray_from_bg = False): #Apply Canny Edge detection on a grayscale image
        #First convert to grayscale, either by background substraction or conventionally
        if gray_from_bg:
            gray = self.bg_substraction()
        else:
            self.rgb = self.org.copy()
            self.gray = cv.cvtColor(self.rgb, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray,(self.gauss_window_size,self.gauss_window_size),0)
            
        #Apply Canny edge detection
        self.bw = cv.Canny(image=gray, threshold1=self.cannysoft, threshold2=self.cannyhard)
        
        return self.bw

    def detect_spots(self): #Detects spots using the BW image
        #Calculate min and max spot radius in px
        self.minradpx = int(np.floor((self.min_spot_r * self.mmtopm) / self.pxtopm))
        self.maxradpx = int(np.ceil((self.max_spot_r * self.mmtopm) / self.pxtopm))
        maxareapx2 = np.square(self.maxradpx) * np.pi
        minareapx2 = np.square(self.minradpx) * np.pi
        
        #Find all enclosed shapes
        contours, hier = cv.findContours(self.bw,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        print("Number of enclosed objects: ",str(len(contours)))
        
        #Revert to original image
        self.rgb = self.org.copy()
        
        #Get centers and calculate area for each spot
        areas = []
        centers = []
        textpos = []
        circularities = []
        cinds = []
        rectangles = []
        #Go through each spot candidate
        for cind, c in enumerate(contours):
            area = cv.contourArea(c)
            #Filter by area
            if area > maxareapx2:
                continue
            if area < minareapx2:
                continue
                
            #Find the smallest bounding rectangle
            rx,ry,rw,rh = cv.boundingRect(c)
            rx1 = max(int(rx),0)
            rx2 = min([(rx + rw),self.width])
            ry1 = max(int(ry),0)
            ry2 = min([(ry + rh),self.height])
            
            #Draw white filled contour on a black dummy rectangle
            sub_bw = np.zeros((rh,rw),np.uint8)
            sub_bw = cv.drawContours(sub_bw, contours, cind, 255,-1,offset=(-rx1,-ry1))
            
            #Draw a dummy ellipse in the bounding rectangle to compare and calculate circularity of the spot/contour
            xdim = sub_bw.shape[0]
            ydim = sub_bw.shape[1]
            subellipse = np.zeros((xdim,ydim),np.uint8)
            subellipse = cv.ellipse(subellipse,(int(ydim/2),int(xdim/2)),(int(ydim/2),int(xdim/2)),0,0,360,255,-1,cv.FILLED,0)
            circularity = np.sum(subellipse == sub_bw) / (xdim * ydim)
            
            #Filter by circularity
            if circularity < self.circ_threshold:
                continue
                
            #Save spot information
            areas.append(area)
            circularities.append(circularity)
            rectangles.append([rx1,ry1,rx2,ry2])
            textpos.append([rx2,ry1])
            
            #Calculate spot's center of mass
            M = cv.moments(c)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            center = (x,y)
            centers.append(center)
            cinds.append(cind)
            
        centers = np.asarray(centers)
        rectangles = np.asarray(rectangles)
        areas = np.asarray(areas)
        textpos = np.asarray(textpos)
        circularities = np.asarray(circularities)
        cinds = np.asarray(cinds)
        
        def get_nearest_center(x,y,centers,threshold):
            dist = centers - np.asarray([x,y])
            dist = np.sum(np.square(dist),1)
            min_index = np.argmin(dist)
            if (dist[min_index] > np.square(threshold)):
                if (x == 0 & y == 0):
                    return min_index
                else:
                    return -1
            else:
                return min_index
            
        def find_spots_on_row(centers,y,threshold,direction,exclusionList):
            #find centers that are within the threshold margin of given y
            spots_on_row = []
            spot_x = []
            sumy = 0
            ymax = y + threshold
            ymin = y - threshold
            found = True
            while found:
                found = False
                for i, c in enumerate(centers):
                    if (c[1] < ymax) & (c[1] > ymin):
                        if i in spots_on_row:
                            continue
                        if i in exclusionList:
                            continue #skip already detected centers
                        found = True
                        spots_on_row.append(i)
                        spot_x.append(c[0])
                        sumy += c[1]
                        y = sumy / len(spots_on_row)
                        ymax = y + threshold
                        ymin = y - threshold
            spot_x = np.asarray(spot_x)
            order = np.argsort(spot_x * direction)
            spots_on_row = np.asarray(spots_on_row)
            spots_on_row = spots_on_row[order]
            return np.asarray(spots_on_row)
            
            
        def serpentine_sort_spots(max_x,max_y,centers,stepsize,threshold):
            #Sort the spots in a serpentine pattern (as in reading a book; left to right, top to bottom)
            sorted_spots = []
            last_y = np.min(centers[:,1])
            direction = 1 #1 to the right, -1 to the left
            while len(sorted_spots) < centers.shape[0]:
                spots_on_row = find_spots_on_row(centers,last_y,threshold,direction,sorted_spots)
                tobedeleted = []
                for s in sorted_spots: #find duplicates and keep already existing one
                    ind = np.where(spots_on_row == s)
                    if len(ind[0] > 0):
                        tobedeleted.append(ind[0])
                spots_on_row = np.delete(spots_on_row,tobedeleted,axis=0)
                #add spots
                sorted_spots.extend(spots_on_row)
                #move to the next row
                if len(spots_on_row) > 0:
                    direction *= -1
                    last_y = np.mean(np.asarray(centers)[spots_on_row,1]) + stepsize
                else:
                    last_y += stepsize
                if last_y > max_y:
                    break;
            return centers[np.asarray(sorted_spots),:],np.asarray(sorted_spots)
          
        #Serpentine sort spots and their corresponding information
        centers,sorted_ind = serpentine_sort_spots(self.width, self.height, centers, int(self.dist_btw_spots * self.mmtopx), self.maxradpx)
        rectangles = rectangles[sorted_ind,:]
        areas = areas[sorted_ind]
        textpos = textpos[sorted_ind,:]
        circularities = circularities[sorted_ind]
        cinds = cinds[sorted_ind]
        
        #Delete small spots that are inside of a bigger spot
        tobedeleted = []
        for i in range(len(rectangles) - 1):
            for j in range(i+1,len(rectangles)):
                if areas[i] > areas[j]:
                    r1 = rectangles[i]
                    r2 = rectangles[j]
                    smaller = j
                else:
                    r1 = rectangles[j]
                    r2 = rectangles[i]
                    smaller = i
                if (r1[0] <= r2[0]) & (r1[2] >= r2[2]) & (r1[1] <= r2[1]) & (r1[3] >= r2[3]):
                    tobedeleted.append(smaller)      
        centers = np.delete(centers,tobedeleted,axis=0)
        rectangles = np.delete(areas,tobedeleted,0)
        areas = np.delete(areas,tobedeleted,0)
        textpos = np.delete(textpos,tobedeleted,0)
        circularities = np.delete(circularities,tobedeleted,0)
        cinds = np.delete(cinds,tobedeleted,0)
        
        #Plot spot edges and numberings on the original image
        for i, c in enumerate(centers):
            center = (c[0], c[1])
            cv.circle(self.rgb, center, int(self._line_width), (0, 255, 0), -1)
            cv.putText(self.rgb, str(i+1), (textpos[i,0],textpos[i,1]), cv.FONT_HERSHEY_SIMPLEX,int(self._font_size), (255, 0, 0),3)
            cv.drawContours(self.rgb, contours, cinds[i], (0,255,0), int(self._line_width))
        
        #Save area info after converting to cm2
        self.areascm2 = areas * self.pxtomm * self.pxtomm / 100
        
        #Generate coordinates for center of each spot (in mm)
        #The first spot is 0,0
        self.coordinates = []
        currentx = centers[0][0]
        currenty = centers[0][1]
        for i, c in enumerate(centers):
            tempx = (c[0] - currentx) * self.pxtomm
            tempy = (currenty - c[1]) * self.pxtomm
            self.coordinates.append([i + 1,tempx,tempy])
            currentx = c[0]
            currenty = c[1]
        self.coordinates = np.array(self.coordinates)
                  
        return self.rgb
    
    def export_teaching_coordinates(self, filepath = None, newfilepath = None):
	    #Generate new teaching coordinates if initial coordinates are given (in um)
        if filepath is None:
            filepath = self.targetdir + "/" + self.fname + '_teaching.csv'
        teaching_file = Path(filepath)
        if ~(teaching_file.is_file()):
            return None
        with open(filepath, 'r') as oldteaching:
              alllines = oldteaching.readlines()
        headers = alllines[0]
        temp = alllines[1]
        params = temp.split(',')
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_newteaching.csv')
        with open(newfilepath, 'w') as newteaching:
            newteaching.write(headers)
            newteaching.write(temp)
            for c in self.coordinates:
                  params[0] = "{:.3f}".format(float(params[0]) + (c[1] * 1000))
                  params[1] = "{:.3f}".format(float(params[1]) + (c[2] * 1000))
                  newteaching.write(','.join(params))
	
    def export_coordinates(self, newfilepath = None):
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_coordinates.txt')
        np.savetxt(newfilepath, self.coordinates, delimiter='\t',comments='',header='\t'.join(['#','X','Y']),fmt=['%d','%.10f','%.10f'])
        
    def export_areas(self, newfilepath = None):
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_areas.txt')
        areas_txt = np.concatenate((np.expand_dims(np.arange(self.areascm2.shape[0]),-1) + 1,np.expand_dims(self.areascm2, -1)),1)
        np.savetxt(newfilepath, areas_txt, delimiter=",",fmt=['%d','%.10f'],header='#,Area(cm2)',comments='')
    
    def export_bw(self, newfilepath = None):
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_bw.png')
        cv.imwrite(newfilepath, self.bw)
        return newfilepath
        
    def export_rgb(self, newfilepath = None):
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_rgb.png')
        cv.imwrite(newfilepath, self.org)
        return newfilepath
		
    def export_gray(self, newfilepath = None):
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_gray.png')
        cv.imwrite(newfilepath, self.gray)
        return newfilepath
    
    def export_rgb_detected(self, newfilepath = None):
        if newfilepath is None:
            newfilepath = str(self.targetdir + "/" + self.fname + '_spots.png')
        cv.imwrite(newfilepath, self.rgb)
        return newfilepath
    
    def start(self): #Start processing VK4 file and detect spots
        print("Started.")
        print("Loading VK4 file...")
        self.load_vk4()
        print("VK4 file loaded.")
        print("Removing background...")
        self.get_rect_mean()
        self.bg_substraction()
        print("Background substracted.")
        self.bw_by_threshold()
        if self.closing_enabled:
            self.morph_closing()
        print("Detecting spots...")
        self.detect_spots()
        print("Number of detected spots: ", str(len(self.coordinates)))
        print("Finished.")
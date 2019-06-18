#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:12:25 2019

@author: ilkay.isik
"""

#%%  INTRO
'''
Workflow to create PFOB ROI's algorithmically 
(using parcels from Julian et al 2012)

This code does:
1. Take individual activation maps for a certain contrast
2. Transform the maps to a common space
3. Threshold the maps at a reasonable level and save a probabilistic map

To DO
4. Parcellate the overlap map by using an image segmentation algorithm
5. Select a set of meaningful parcels
6. Define subject specific ROI's by intersecting each meaningful parcel with 
each individual thresholded activation map
7. Define the ROI's based on this


dependencies: antsApplyTransforms
'''
# %% IMPORTS and PATHS
from glob import glob
import os
from nipype.interfaces.ants import ApplyTransforms
from nibabel import load, save, Nifti1Image
from nilearn import plotting
import numpy as np
from nistats.thresholding import map_threshold
import matplotlib.pyplot as plt
#%%
root = '/Users/ilkay.isik/project_folder_temp/fc_content/MRI_data/lscp_data/derivatives/'
ants_files = sorted(glob(root + 'fmriprep/sub-*/anat/sub-*_T1w_target-'\
                               'MNI152NLin2009cAsym_warp.h5'))
nr_sub = len(ants_files) 
mni_temp = root + 'mni152.nii'

#%% 1. Take individual activation maps for a certain contrast 
# PLACE
p_zmap = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat13.nii.gz'))
# FACE
f_zmap = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat12.nii.gz'))
# OBJECT
o_zmap = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat11.nii.gz'))
# BODY
b_zmap = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat14.nii.gz'))

pfob = [p_zmap, f_zmap, o_zmap, b_zmap]
names = ['place_prob', 'face_prob', 'object_prob', 'body_prob']

#%%2. Transform the maps to a common space [3x3x3 MNI]
mni_temps =  sorted(glob(root + '/fmriprep/sub-*/func/sub-*_task-pfobloc_'\
                         'run-01_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'))
for loc in pfob:
    for c, zmap in enumerate(loc):
        norm = ApplyTransforms(reference_image=mni_temps[c],
                               input_image_type=3,
                               float=True,
                               interpolation='Linear',
                               invert_transform_flags=[False],
                               output_image=zmap.replace('.nii.gz', '_inMNI_3mm.nii.gz'),
                               input_image=zmap,
                               transforms=ants_files[c])
        res = norm.run()
        print(c+1, zmap)
# %% Load the normalized maps
p_norm = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat13_inMNI_3mm.nii.gz'))
f_norm = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat12_inMNI_3mm.nii.gz'))
o_norm = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat11_inMNI_3mm.nii.gz'))
b_norm = sorted(glob(root + 'analysis/sub-*/pfobloc/pfobloc.feat/stats/zstat14_inMNI_3mm.nii.gz'))
pfob_norm = [p_norm, f_norm, o_norm, b_norm]

# %% 3. Threshold and binarize the maps at a reasonable level
# Save probabilistic maps and images out

h, thr ='fdr', 0.05 # thresholding parameters
prob_maps ='/Users/ilkay.isik/MPI-Documents/hack_19/data/prob_maps2'

if not os.path.exists(prob_maps):
    os.makedirs(prob_maps)

for c, loc in enumerate(pfob_norm):
    zmaps = [load(zmap) for zmap in loc]
    thr_images, thresholds = [], []
    
    for i in range(nr_sub):
        thr_img, threshold = map_threshold(zmaps[i], level=thr, height_control=h)
        thr_images.append(thr_img)
        thresholds.append(threshold)
    
    data = [zmap.get_data() for zmap in thr_images]
    bin_data = []
    for i  in range(nr_sub):
        bin_data.append(np.where(data[i] > thresholds[i], 1, 0))

    # sum the images
    sum_img = np.sum(bin_data, axis=0)
    # save out
    out = Nifti1Image(sum_img, header=zmaps[0].header, affine=zmaps[0].affine)
    save(out, prob_maps + '/' + names[c] + "sum.nii.gz")
    
    # plots
    plotting.plot_stat_map(out, bg_img=mni_temp)
    plt.savefig(prob_maps + '/' + names[c] + '_sum_plot.png', dpi=300)
    plt.close()
    
    plotting.plot_glass_brain(out, black_bg=True, display_mode='lyrz', colorbar=True,
    title=names[c])
    plt.savefig(prob_maps + '/' + names[c] + '_sum_glass_summary.png', dpi=300)
    plt.close()
    
    # threshold the sum image and save out as a mask
    thr_sum = np.where(sum_img > 6, 1, 0)
    
    out = Nifti1Image(thr_sum, header=zmaps[0].header, affine=zmaps[0].affine)
    save(out, prob_maps + '/' + names[c] + "_thr_sum.nii.gz")
    
    # plots
    plotting.plot_stat_map(out, bg_img=mni_temp, title=names[c])
    plt.savefig(prob_maps + '/' + names[c] + '_thr_sum.png', dpi=300)
    plt.close()

    plotting.plot_glass_brain(out, black_bg=True, display_mode='lyrz', colorbar=True,
    title=names[c])
    plt.savefig(prob_maps + '/' + names[c] + '_thr_sum_summary.png', dpi=300)
    plt.close()


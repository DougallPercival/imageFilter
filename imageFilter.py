"""
This function provides a method to add image filtering (lightening or darkening) to an image with Python.
Works best with Open CV.
"""

import cv2
import numpy as np
import copy


def lightFiltering(_img, mode=0, method=0, solid_pct=0, grad_low_pct=0,grad_high_pct=100, grad_axis=0, grad_start=0, img_type='uint8'):
    """
    Apply filtering to an image, i.e. darkening/shdaing or lightening an image.
    Inputs:
    _img - the image to apply this to. Returns a new copy 
    mode - 0/1, brighten/darken
    method - 0/1, solid or gradient
    solid_pct - if method selected is solid, defines the percentage to lighten/darken by
    grad_low_pct - if method selected is gradient, defines the low percentage of the gradient to darken/lighten by
    grad_high_pct - if method selected is gradient, defines the high percentage of the gradient to darken/lighten by
    grad_axis - 0 for horizontal(default), 1 for vertical
    grad_start - 0/1. If 0, start at left/top of image. If 1, start at right/bottom of the image, depending on axis.
    img_type - default uint8 - integer type of the image (recommended to use the default uint8)
    """
    # error handling
    if mode not in [0, 1]:
        if str(mode).upper() not in ['BRIGHTEN', 'DARKEN']:
            raise ValueError('lightFiltertering: mode must be a value of 0/brighten or 1/darken')
    if method not in [0, 1]:
        if str(method).upper() not in ['SOLID', 'GRADIENT']:
            raise ValueError('lightFiltertering: method must be a value of 0/solid or 1/gradient')
    if grad_axis > 1 or grad_axis < 0:
        raise ValueError('lightFiltertering: grad_axis must be a value of 0 (horizontal) or 1 (vertical)')
    if grad_start > 1 or grad_start < 0:
        raise ValueError('lightFiltertering: grad_start must be a value of 0 (top/left) or 1 (bottom/right)')

    img = copy.deepcopy(_img)
    
    """
    Solid
    """
    # if solid, just add/subtract from all channels
    if method == 0:

        
        # if percentage is 0, do not apply anything
        if solid_pct == 0:
            solid_pct = 0.0001
        
        # lambda function for updating brightness/darkness
        solid_update_b = lambda x: min(255, x + (x * (solid_pct/100)))
        solid_update_d = lambda x: max(0, x + (x * (solid_pct/100)*-1))
            
        if mode == 0:
            update = np.vectorize(solid_update_b)
        elif mode == 1:
            update = np.vectorize(solid_update_d)
    
        upd_img = update(img)

        
    """
    Gradient
    """
    # if gradient, needs a more complex approach  
    if method == 1:
        
        # create a numpy array with same shape, but with percentages from low-high in selected order
        orig_shape = img.shape
        
        # define  values for height (h), width (w)
        h = orig_shape[0]
        w = orig_shape[1]
        
        # determine which direction to grade on
        # g is the gradient number - when writing the value incrementation add this here
        # o_g means off gradient, and is the other value
        if grad_axis == 0:
            g = h
            o_g = w
        else:
            g = w
            o_g = h
        
        # define the grid of multiplicable numbers
        # if grad_start is at 0, lower - higher. If at 1, higher - lower
        # also create the grid of multiplicable numbers for the gradient step
        if grad_start == 0:
            grad_inc = (grad_high_pct-grad_low_pct)/(o_g)
            grad_grid = np.mgrid[grad_low_pct:grad_high_pct:grad_inc]
        elif grad_start == 1:
            grad_dec = (grad_low_pct-grad_high_pct)/(o_g)
            grad_grid = np.mgrid[grad_high_pct:grad_low_pct:grad_dec]
            
        # the above grid is a shape of (w or h, 1)
        # we must expand to form a shape of (h, w)
        # horizontal
        if grad_axis == 0:
            grad_grid = np.tile(grad_grid,(g,1))
        #vertical
        elif grad_axis == 1:
            _grads = []
            for i in range(g):
                _grads.append(grad_grid)
            _grads = tuple(_grads)
            gr = np.stack(_grads)
            grad_grid = gr.T
        
        #update the shape so it is broadcastable to the lambda
        grad_grid = grad_grid.reshape((h,w,1))
        
        #define lambdas for updating values - x is from the 
        grad_update_b = lambda x, y: min(255, x + (x * (y/100)))
        grad_update_d = lambda x, y: max(0, x + (x * (y/100)*-1))
        if mode == 0:
            update = np.vectorize(grad_update_b)
        elif mode == 1:
            update = np.vectorize(grad_update_d)
        # now multiply this grid with the original image
        upd_img = update(img, grad_grid)
        
        
    # final output of the image        
        
	# only mess with this if you know what you are doing.
    if img_type == 'uint8':
        upd_img = upd_img.astype(np.uint8)
    return upd_img
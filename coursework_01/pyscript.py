# %% [markdown]
# # Coursework 1: Image filtering
# 
# In this coursework you will practice techniques for image filtering. The coursework includes coding questions and written questions. Please read both the text and the code in this notebook to get an idea what you are expected to implement.
# 
# ## What to do?
# 
# * Complete and run the code using `jupyter-lab` or `jupyter-notebook` to get the results.
# 
# * Export (File | Save and Export Notebook As...) the notebook as a PDF file, which contains your code, results and answers, and upload the PDF file onto [Scientia](https://scientia.doc.ic.ac.uk).
# 
# * Instead of clicking the Export button, you can also run the following command instead: `jupyter nbconvert coursework_01_solution.ipynb --to pdf`
# 
# * If Jupyter complains about some problems in exporting, it is likely that pandoc (https://pandoc.org/installing.html) or latex is not installed, or their paths have not been included. You can install the relevant libraries and retry. Alternatively, use the Print function of your browser to export the PDF file.
# 
# * If Jupyter-lab does not work for you at the end (we hope not), you can use Google Colab to write the code and export the PDF file.
# 
# ## Dependencies:
# 
# You need to install Jupyter-Lab (https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) and other libraries used in this coursework, such as by running the command:
# `pip3 install [package_name]`
# 

# %%
# Import libaries (provided)
import imageio.v3 as imageio
import numpy as np
import matplotlib.pyplot as plt
import noise
import scipy
import scipy.signal
import math
import time

# %% [markdown]
# ## 1. Moving average filter (20 points).
# 
# Read the provided input image, add noise to the image and design a moving average filter for denoising.
# 
# You are expected to design the kernel of the filter and then perform 2D image filtering using the function `scipy.signal.convolve2d()`.

# %%
# Read the image (provided)
image = imageio.imread('campus_snow.jpg')
plt.imshow(image, cmap='gray')
plt.gcf().set_size_inches(8, 8)

# %%
# Corrupt the image with Gaussian noise (provided)
image_noisy = noise.add_noise(image, 'gaussian')
plt.imshow(image_noisy, cmap='gray')
plt.gcf().set_size_inches(8, 8)

# %% [markdown]
# ### Note: from now on, please use the noisy image as the input for the filters.
# 
# ### 1.1 Filter the noisy image with a 3x3 moving average filter. Show the filtering results.

# %%
# Design the filter h
### Insert your code ###
h = np.full((3,3),1/9)

# Convolve the corrupted image with h using scipy.signal.convolve2d function
### Insert your code ###
image_filtered = scipy.signal.convolve2d(image_noisy,h,mode = 'same')

# Print the filter (provided)
print('Filter h:')
print(h)

# Display the filtering result (provided)
plt.imshow(image_filtered, cmap='gray')
plt.gcf().set_size_inches(8, 8)

# %% [markdown]
# ### 1.2 Filter the noisy image with a 11x11 moving average filter.

# %%
# Design the filter h
### Insert your code ###
h = np.full((11,11),1/121)

# Convolve the corrupted image with h using scipy.signal.convolve2d function
### Insert your code ###
image_filtered = scipy.signal.convolve2d(image_noisy,h,mode = 'same')

# Print the filter (provided)
print('Filter h:')
print(h)

# Display the filtering result (provided)
plt.imshow(image_filtered, cmap='gray')
plt.gcf().set_size_inches(8, 8)

# %% [markdown]
# ### 1.3 Comment on the filtering results. How do different kernel sizes influence the filtering results?

# %% [markdown]
# ### Insert your answer ###
# 
# Kernel sizing affects the filtering by increasing the number of terms in the moving average filter. As you increase the terms, you change the shape of the frequency response, decreasing the cutoff frequency and transition band size. It also reduces the stopband ripple. In terms of visual changes, it smooths out the image even more, as the kernel size approaches the image size (K->N) then it the image will appear as one colour (intensity for gray scale).

# %% [markdown]
# ## 2. Edge detection (56 points).
# 
# Perform edge detection using Sobel filtering, as well as Gaussian + Sobel filtering.

# %% [markdown]
# ### 2.1 Implement 3x3 Sobel filters and convolve with the noisy image.

# %%
# Design the filters
### Insert your code ###
sobel_x = np.array(([1,0,-1],[2,0,-2],[1,0,-1]))
sobel_y = np.array(([1,2,1],[0,0,0],[-1,-2,-1]))

# Image filtering
### Insert your code ###
g_x = scipy.signal.convolve2d(image_noisy,sobel_x,mode = 'same')
g_y = scipy.signal.convolve2d(image_noisy,sobel_y,mode = 'same')



# Calculate the gradient magnitude
### Insert your code ###
grad_mag = np.sqrt(np.square(g_x)+np.square(g_y))


# Print the filters (provided)
print('sobel_x:')
print(sobel_x)
print('sobel_y:')
print(sobel_y)

# Display the magnitude map (provided)
plt.imshow(grad_mag, cmap='gray')
plt.gcf().set_size_inches(8, 8)

# %% [markdown]
# ### 2.2 Implement a function that generates a 2D Gaussian filter given the parameter $\sigma$.

# %%
# Design the Gaussian filter
def gaussian_filter_2d(sigma):
    # sigma: the parameter sigma in the Gaussian kernel (unit: pixel)
    #
    # return: a 2D array for the Gaussian kernel
    
    ### Insert your code ###
    k = 3
    ksigma = k*sigma

    x = y = np.linspace(-ksigma,ksigma,num=2*ksigma + 1)
    
    normal = 1/(2* np.pi * np.square(sigma))
    xv, yv = np.meshgrid(x,y)
    h = normal * np.exp(-1*(np.square(xv) + np.square(yv))/(2*np.square(sigma)))
    return h


"""
def loopfilter(sigma):
    norm = 1/(2*3.14*np.square(sigma))
    h = np.zeros((3*2*sigma,3*2*sigma))
    for x in range(-3 * sigma , 3*sigma):
        for y in range(-3 * sigma , 3*sigma):
            xc = x+15
            yc = y+15
            h[xc,yc] = norm * np.exp(-1*(np.square(x)+np.square(y))/(2*np.square(sigma)))
    return h
"""
# Visualise the Gaussian filter when sigma = 5 pixel (provided)
sigma = 5
h = gaussian_filter_2d(sigma)
#hl = loopfilter(sigma)
plt.imshow(h)
#plt.imshow(hl)

# %% [markdown]
# ### 2.3 Perform Gaussian smoothing ($\sigma$ = 5 pixels) and evaluate the computational time for Gaussian smoothing. After that, perform Sobel filtering and show the gradient magintude map.

# %%
# Construct the Gaussian filter
### Insert your code ###
h = gaussian_filter_2d(5)

# Perform Gaussian smoothing and count time
### Insert your code ###
start = time.time()

f = scipy.signal.convolve2d(image_noisy, h,mode = 'same')

stop = time.time()
print(stop - start)

# Image filtering 
### Insert your code ###

g_x = scipy.signal.convolve2d(f,sobel_x,mode = 'same')
g_y = scipy.signal.convolve2d(f,sobel_y,mode = 'same')


# Calculate the gradient magnitude
### Insert your code ###

grad_mag = np.sqrt(np.square(g_x)+np.square(g_y))

# Display the gradient magnitude map (provided)
plt.imshow(grad_mag, cmap='gray', vmin=0, vmax=100)
plt.gcf().set_size_inches(8, 8)

# %% [markdown]
# ### 2.4 Implement a function that generates a 1D Gaussian filter given the parameter $\sigma$. Generate 1D Gaussian filters along x-axis and y-axis respectively.

# %%
## Design the Gaussian filter
def gaussian_filter_1d(sigma):
    # sigma: the parameter sigma in the Gaussian kernel (unit: pixel)
    #
    # return: a 1D array for the Gaussian kernel
    k = 3
   
    ### Insert your code ###
    normal = 1/(np.sqrt(2* np.pi) * sigma)
    n = np.linspace(-k*sigma,k*sigma,2*k*sigma + 1)
    #print(n)
    h = normal * np.exp(-1*((np.square(n))/(2*np.square(sigma))))
    h = h.reshape(-1,1)
    return h

# sigma = 5 pixel (provided)
sigma = 5

# The Gaussian filter along x-axis. Its shape is (1, sz).
### Insert your code ###
h_x = np.transpose(gaussian_filter_1d(sigma))

# The Gaussian filter along y-axis. Its shape is (sz, 1).
### Insert your code ###
h_y = gaussian_filter_1d(sigma)
#print(h_y)

# Visualise the filters (provided)
plt.subplot(1, 2, 1)
plt.imshow(h_x)
plt.subplot(1, 2, 2)
plt.imshow(h_y)

# %% [markdown]
# ### 2.6 Perform Gaussian smoothing ($\sigma$ = 5 pixels) using two separable filters and evaluate the computational time for separable Gaussian filtering. After that, perform Sobel filtering, show the gradient magnitude map and check whether it is the same as the previous one without separable filtering.

# %%
# Perform separable Gaussian smoothing and count time
### Insert your code ###
gaus_x = gaussian_filter_1d(5).transpose()
print(gaus_x)
gaus_y = gaussian_filter_1d(5)

#gaus = scipy.signal.convolve2d(gaus_x,gaus_y)

# Image filtering
### Insert your code ###

#gaus
st =time.time()

f = scipy.signal.convolve2d(image_noisy,gaus_x, mode='same')
f = scipy.signal.convolve2d(f,gaus_y,mode = 'same')
print(time.time()-st)
plt.imshow(f, cmap='gray')
plt.gcf().set_size_inches(8, 8)
print(f.shape)

g_x = scipy.signal.convolve2d(f,sobel_x,mode = 'same')
g_y = scipy.signal.convolve2d(f,sobel_y,mode = 'same')

# Calculate the gradient magnitude
### Insert your code ###
grad_mag2 = np.sqrt(np.square(g_x)+np.square(g_y))

# Display the gradient magnitude map (provided)
#plt.imshow(grad_mag2, cmap='gray', vmin=0, vmax=100)
#plt.gcf().set_size_inches(8, 8)
# Check the difference between the current gradient magnitude map
# and the previous one produced without separable filtering. You
# can report the mean difference between the two.
### Insert your code ###

differ = np.abs(grad_mag2 - grad_mag)
meandiff = np.mean(differ)
print(meandiff)


# %% [markdown]
# ### 2.7 Comment on the Gaussian + Sobel filtering results and the computational time.

# %% [markdown]
# ### Insert your answer ###
# Gaussian filtering is needed in order to achieve gaussian smoothing. This is so that minute, sharp changes in amplitude due to noise are less impactful on the derivative functions. These derivative functions in discrete time can be generated via the centre difference. The centre differences are calculated using the Sobel/Prewitt filters respective to both the x and y axis. Sobel/Prewitt filters have a lowpass filter in the perpendicular direction of their difference equations, this aids in the ability to detect edges by smoothing the noise in the complementary direction. The visual result is an image with the horizontal and vertical edges highlighted with a higher intensity whilst the other points in the image have their intensity reduced almost to black pixels. The output of the two Sobel filters are combined using the gradient magnitude equation which creates the image you see above.
# 
# In the frequency domain, the effect of the cascading filters is that of a low pass filter followed by a high pass filter.
# 
# With regards to the computational time, a Sobel Filter has K^2 multiplications and K^2 -1 summations and this happens for N^2 pixels. Therefore, the complexity is O(N^2 * K^2. Gaussian Kernel also has the same complexity, however, both of these are seperable. Once seperated, the complexity is O(N^2 * K) due to 2N^2 * K multiplications and 2N^2 * (K-1) summations.

# %% [markdown]
# ## 3. Challenge: Implement 2D image filters using Pytorch (24 points).
# 
# [Pytorch](https://pytorch.org/) is a machine learning framework that supports filtering and convolution.
# 
# The [Conv2D](https://pytorch.org/docs/stable/nn.html#conv2d) operator takes an input array of dimension NxC1xXxY, applies the filter and outputs an array of dimension NxC2xXxY. Here, since we only have one image with one colour channel, we will set N=1, C1=1 and C2=1. You can read the documentation of Conv2D for more detail.

# %%
# Import libaries (provided)
import torch


# %% [markdown]
# ### 3.1 Expand the dimension of the noisy image into 1x1xXxY and convert it to a Pytorch tensor.

# %%
# Expand the dimension of the numpy array
### Insert your code ###
fourdimage = image_noisy[np.newaxis,np.newaxis,:,:]
print (fourdimage.shape)

# Convert to a Pytorch tensor using torch.from_numpy
### Insert your code ###
imgTensor = torch.from_numpy(fourdimage)


# %% [markdown]
# ### 3.2 Create a Pytorch Conv2D filter, set its kernel to be a 2D Gaussian filter and perform filtering.

# %%
# A 2D Gaussian filter when sigma = 5 pixel (provided)
sigma = 5
h = gaussian_filter_2d(sigma)
htensor = torch.from_numpy(h[np.newaxis,np.newaxis,:,:])

# Create the Conv2D filter
### Insert your code ###
nn = torch.nn.Conv2d(1,1,kernel_size=h.shape,stride=1,padding="same")
nn.weight.data = htensor.float()
# Filtering
### Insert your code ###
out = nn(imgTensor.float())

# Display the filtering result (provided)
#plt.imshow(image_filtered, cmap='gray')
#plt.gcf().set_size_inches(8, 8)

plt.imshow(out.detach().numpy().squeeze(), cmap='gray')
plt.gcf().set_size_inches(8, 8)

print(np.mean(np.abs(out.detach().numpy().squeeze() - f)))

# %% [markdown]
# ### 3.3 Implement Pytorch Conv2D filters to perform Sobel filtering on Gaussian smoothed images, show the gradient magnitude map.

# %%
# Create Conv2D filters
### Insert your code ###
sigma = 5
gaussianfilter = torch.from_numpy(gaussian_filter_2d(sigma)[np.newaxis,np.newaxis,:,:])

sobeltensorx = torch.from_numpy(sobel_x[np.newaxis,np.newaxis,:,:])
sobeltensory = torch.from_numpy(sobel_y[np.newaxis,np.newaxis,:,:])

nn_sobelx = torch.nn.Conv2d(1,1,kernel_size=sobel_x.shape,stride=1,padding="same")
nn_sobely = torch.nn.Conv2d(1,1,kernel_size=sobel_y.shape,stride=1,padding="same")
# Perform filtering
### Insert your code ###
nn_sobelx.weight.data = sobeltensorx.float()
outx = nn_sobelx(out)
nn_sobely.weight.data = sobeltensory.float()
outy = nn_sobely(out)
# Calculate the gradient magnitude map
### Insert your code ###
grad_mag3 = np.sqrt(np.square(outx.detach().numpy()) + np.square(outy.detach().numpy())).squeeze()
# Visualise the gradient magnitude map (provided)
plt.imshow(grad_mag3.squeeze(), cmap='gray', vmin=0, vmax=100)
plt.gcf().set_size_inches(8, 8)
print(np.mean(np.abs(grad_mag3 - grad_mag2)))

# %%



# Want to do canny edge detection now, first step is to filter using gaussian filter

print(grad_mag2)


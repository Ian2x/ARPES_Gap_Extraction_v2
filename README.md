# ARPES_Gap_Extraction_v2

File explanations:

 - data_reader.py
Reads in data files. Can zoom in to a portion of the spectrum. Can also symmetrize data across w=0.

 - extract_ac.py
The main fitting function. Fits each EDC to get a peak location. By default, fits the k=0 EDCs first, then expands outward (stage 1). Then, using each EDC's peak location, fits a curve to the peak locations (stage 2). A lot of effort was put into improving the fit by adjusting which EDC peaks are used in stage 2. Typically, using a wider range of EDCs results in a lower gap size. Using a narrower range of EDCs results in a higher gap size.

 - extract_k_dependent.py
Leftover code from when a full 2D fit was used. A polynomial fit was used to model the evolution of various parameters as a function of k.

 - extraction_functions.py
Various fitting models. Features various papers, convolutions, secondary electron models, symmetrizations, etc. Also has functions to exclude high-noise data, symmetrize data, etc.

 - figures.py
Function for creating figures for potential paper.

 - fitter.py
This class used to do more, but now is only used to do a standard EDC fit. It has different models for real vs. simulated data.

 - general.py
A bunch of random functions (e.g. energy convolution, fermi-dirac, statistical tests, etc.)

 - main.py
Main function. All other functions are called from here.

 - simulation.py
Class for generating simulated spectrum.

 - spectral_functions.py
Models for generating simulated spectrum.

Problems:

1) In stage 1, there are a wide range of viable peak locations, especially when there are overlapping peaks. This means that the starting peak estimate has a big effect on the output. For instance, fitting the EDCs from center-out versus outside-in makes a big difference. This problem can be imagined as each EDC having a range of viable peak locations, but in reality the fit only outputs one point for each EDC. Because each EDC fit also serves as the starting guess for the next EDC fit, this effect compounds. I also tried fitting each EDC independently, but this led to very erratic peak estimates (as the EDCs move further from k=0 and closer to kf). 
 - I tried dealing with the problem by adding more complex ways of choosing the range of EDCs that are used in the stage 2 part of the fit. However, this only helped a bit, and doesn't really address the fundamental issue above.







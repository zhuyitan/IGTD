# Image Generator for Tabular Data (IGTD): Converting Tabular Data into Images for Deep Learning Using Convolutional Neural Networks

Image Generator for Tabular Data (IGTD) is an algorithm for transforming tabular data into images for subsequent deep learning analysis using CNNs. The algorithm assigns each feature to a unique pixel in the image representation. Similar features are assigned to neighboring pixels, while dissimilar features are assigned to pixels that are far apart. According to the assignment, an image is generated for each sample, in which the pixel intensity reflects the value of the corresponding feature in the sample. The manuscript of IGTD is currently under review. A link to the paper will be provided later.

Scripts folder includes two scripts, IGTD_Functions.py and Example_Run.py. IGTD_Functions.py provides all the functions used by the IGTD algorithm. Comments explaining the input and output of each function are provided in the script. Example_Run.py gives examples showing how to run the IGTD algorithm for demo purpose.

Data folder includes a small dataset used by the Example_Run.py script for running the demos.

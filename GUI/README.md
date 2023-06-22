#### Download

current version (Nov 8, 2012) : 
[[windows (32-bit)](https://drive.google.com/file/d/0B8KLKywXy-wKbk9TLWdCOXMxV2c/view?usp=sharing)]
[[linux (32-bit)](https://drive.google.com/file/d/0B8KLKywXy-wKSDNoM1NrRXg4dzg/view?usp=sharing)]
[[linux (64-bit)](https://drive.google.com/file/d/0B8KLKywXy-wKZ0c1ZWZFWHJsSlU/view?usp=sharing)]


#### Documentation

- [options](#options) : contains an explanation of input options
- [format](#format) : specifies the input data format
- [template](files/template.xls) : use this to prepare your dataset for analysis
- [output](#output) : describes each output file produced by the software
- [faq](#faq) : frequently asked questions (and answers)


#### Description


![ ](files/glee_gui.jpg)

GLEE conducts a statistical test to identify differentially expressed proteins using (normalized adjusted) spectral count data. 
 
To get started:

1. Download the glee.exe file for your operating system from the links above. 

2. Put the downloaded file in a directory for which you have user-level write privileges, such as your "My Documents" folder.

3. Double-click the glee.exe file to open up the user interface (GUI). No other installation steps are needed. The software comes with all dependencies packaged into the executable, and this makes starting up a bit slow. So, it might take up to a minute for the GUI to show up when you first start it. 

4. Select your input file, set the options and run!  See the "Documentation" section above for guidance on input parameters and how to format your input data file.


#### Credits

Anton Poliakov proposed GLEE's non-linear model for the mean-variance relationship observed in proteomic spectral count data. Lalit Ponnala developed the software using Python and its associated libraries such as Tkinter, Numpy, Scipy and PyInstaller.


#### Contact

If you run into issues with the software or have comments for improving it, contact Lalit Ponnala [lalit dot p at gmail dot com]. 
If you have questions related to proteomics data and differential expression, contact Anton Poliakov [poliakov at med dot umich dot edu]. 

<hr>

#### <a name="options"></a>Input options

- file_name = this is the name of the excel spreadsheet, use the button to select it on your computer
- num_replicates_A = this is the number of replicates in condition A, i.e. the first set of replicates you specify
- num_replicates_B = this is the number of replicates in condition B, i.e. the second set of replicates
- fit_type = this specifies the type of model that will be fit, can be either linear or cubic
- num_iterations = this specifies the number of iterations for re-sampling (recommended value = 10000)
- bin_choice = this specifies the type of binning, can be either equal or adaptive
- num_bins = this is the number of bins into which the signal-range will be divided, the number of proteins in each bin will depend on the choice of binning (recommended value = 20)
- merge_low = the low-signal bins will be merged so that the lowest one contains atleast this percentage of proteins, applies only if adaptive binning is selected
- merge_high = the high-signal bins will be merged so that the highest one contains atleast this percentage of proteins, applies only if adaptive binning is selected
- fit_quantile = this specifies the quantile that will be used in each bin to select the data points to fit the model (recommended value = 0.5, which represents the median)
- output_id = this specifies the name that will be attached to all output files


#### <a name="format"></a>Input data format

- All the data must be in the first sheet of the Excel workbook
- Column headers MUST be specified
- The first column must contain alpha-numeric protein names (cannot accept just numbers!)
- Replicates for the first condition (condition A) must appear after the first column, each replicate in a separate column
- Then the second condition (condition B) replicates must follow in succeeding columns
- No blank columns are allowed between replicates or between conditions
- No blank rows or cells are allowed in the data (please enter 0 for undetected proteins)

<hr>

#### <a name="output"></a>Output

GLEE produces a total of 9 output files that deliver all information pertaining to the analysis. In the following notes, "outname" indicates the name that will be attached to all output files, as specified in the input field "output_id". When the analysis is complete, a message will appear below the RUN button indicating the full path to the location of the output files.

**Text files**

outname.DEG.txt

- this is the main output file containing the results of the differential expression test in the last column
- the remaining columns show the mean, standard deviation, p-value and signal-to-noise ratio
- for easier viewing, copy the contents of this file to an excel spreadsheet

outname.selected_points.txt

- this file (created only if binning is chosen) shows the coordinates that were used to fit the specified model

outname.log.txt

- this file tracks the data analysis steps as they are conducted
- shows model-fit statistics (adjusted R-squared) and the number of proteins found to be differentially expressed


**Image files**

- outname.condition-A.png : shows the raw data and model-fit in condition A (the first condition)
- outname.condition-A.siglevel.png : shows the sorted average expression level in condition A
- outname.condition-B.png : shows the raw data and model-fit in condition B (the second condition)
- outname.condition-B.siglevel.png : shows the sorted average expression level in condition B
- outname.stn_distr.png : shows a histogram of the signal-to-noise (STN) distribution
- outname.stn_pvalue.png : shows the STN values against the calculated p-values 

<hr>

#### <a name="faq"></a>Frequently asked questions (and answers)

Q: I double-clicked the downloaded file, but nothing happens
<br>
A: Patience! It takes about 1-2 minutes for the software to properly load-up and get started. You should be seeing the user-interface pop up soon.


Q: I waited 10 minutes but the software still didn't start up
<br>
A: If you're on windows, please re-start your computer and then re-start GLEE. Doing that helped at least one frustrated user, who's now happily back to running GLEE again.


Q: How long does the program take to complete the analysis?
<br>
A: It depends on the number of proteins you have in your dataset and the number of iterations you specify. During our tests, we used 10,000 iterations on 2000 proteins and it finished in about 2 minutes.


Q: How many iterations should I use? Are 1000 iterations enough?
<br>
A: We recommend using 10,000 iterations. This is usually enough to simulate a fairly stable STN distribution if you have roughly 2000 to 3000 proteins. If you have more than 5000 proteins, you could reduce the number of iterations to 1000, to ensure that the program finishes running in a reasonable time (about 5 minutes).

 
Q: Why don't I get the same p-values when I re-run the program on the same dataset with exactly the same parameters?
<br>
A: Since a randomization is performed on each protein, the simulated STN distribution will differ slightly, leading to slight differences in the p-values even when the software is re-run with identical parameters. But the general trends in the results can be expected to be the same from one run to another, i.e. the proteins that have low p-value (which appear at the top of the output file) should continue to have low p-values when re-run with the same parameters. One way to ensure fairly similar p-values from repeated runs is to increase the number of randomizations, say to 10,000 or even 25,000. That would make the resampled STN distribution fairly stable, leading to "close enough" p-values each time the software is run with the same parameters. 


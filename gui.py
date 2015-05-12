from __future__ import division
from Tkinter import Tk, StringVar, DoubleVar, IntVar, Label, Button, Entry, Frame, Radiobutton, Checkbutton
from tkFileDialog import askopenfilename
from os import getcwd, path
from webbrowser import open_new
from tkMessageBox import showerror
import re
import xlrd
import sys
import numpy as np
from scipy.stats.mstats import mquantiles
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime


# ---- COMMENTS ----

# -- from Dom --
# I was processing the P3 leaf dataset results I got from you and writing the manuscript for it. I compared the list I got of significantly 
# expressed proteins with that from another spectral counting software and I noticed that the main difference lies with proteins with zero SPCs 
# in one or more samples. Is there a way to include another column in the GLEE output to indicate which proteins had missing values and had 
# imputations? Or maybe this is too much.

# print out progress status to the log file, as the messages are being generated
# change the number of iterations internally if the user-specified value is too high - notify the user in log output file
# speed up the code:
	# change the p-value calculation to sort the stn distribution and use first exceeding value
	# ways to use lesser memory if 10,000 iterations are used
# make the right end of the filename appear in the entry box
# right-justify all the labels - set justify for all rows in the first column of the grid? (all labels)
# create output ID from input file name (e.g. Cooper.xls to Cooper-GLEE.*)
# put up a dialog saying "GLEE is starting" so the user knows to wait
	# can a root layout be replaced by another one?

# [done] get rid of FDR option, present output in increasing order of p-value
# [done] create the option to avoid binning the data
# [done] write code to fit the linear/cubic model to un-binned data
# [done] make the default number of iterations 1000
# [done, pre-defined size] set limits on how much the window can be maxmimized
# [done, set padx,pady] center the widgets within the root (parent) widget so they appear nice even when maximized
# [done] make bin choice a checkbox grid
# [done] make fit type a checkbox grid
# [done] change submit button text to RUN
# [doesn't help] import only what's needed from each module so final exe is smaller
# [done] reset button to clear options to their defaults
# [done] help button to show dialog with clickable link to my glee homepage
# [worked around] 2-row elements should show two blank rows, not one!
# [done] create a dialog to display errors after validating input file and options
# [done] check if the entry box for each option contains the specified type (no mix-ups allowed, e.g. no char in a int box, etc), catch errors and show them in a dialog
# [done] write all on-screen comments to a logfile
# [done] all sys.exit commands need to throw exceptions, not just kill the program
# [done, set matplotlib.use('agg')] avoid using the forceful method to keep the gui from closing
# [done] if filename is empty, specify that the user needs to select a file
# [no need, condition order will be retained] ask the user to specify condition names so they can then be printed after model-fit and in the output .DEG file
# -------------------


# set a bunch of default values
filename_default=''
nA_default=0
nB_default=0
p_default=20
fbL_default=0
fbH_default=0
fQ_default=0.5
niter_default=1000
outID_default='output'


# function to read the excel spreadsheet
def xlsread(filename,nA,nB):
	wb=xlrd.open_workbook(filename)
	sh=wb.sheet_by_index(0)
	# check if the number of columns are correct
	if sh.ncols != 1+nA+nB:
		raise Exception('incorrect number of columns in input dataset')
	# first column contains protein names
	P=sh.col_values(0,1,)
	A=[]
	for j in range(1,1+nA):
		A.append(sh.col_values(j,1,))
	B=[]
	for j in range(1+nA,1+nA+nB):
		B.append(sh.col_values(j,1,))
	return P,np.column_stack(A),np.column_stack(B)


# function to fit straight line and polynomial
def regress(x,y,order):
	if (len(x)!=len(y)):
		raise Exception('regress: x and y are of unequal length')
	nobs=len(x); ncoef=order+1;
	fit=np.polyfit(x,y,order,full=True)
	coefs = fit[0]
	SSE = fit[1][0]
	SST = sum((y - np.mean(y))**2)
	Rsq = (SST - SSE)/SST
	adjRsq = 1-(1-Rsq)*((nobs-1)/(nobs-ncoef))
	return coefs,Rsq,adjRsq


# function to calculate model-based standard-deviation
def calcs(x,coeff):
	if np.any(x==0):
		raise Exception('calcs: x contains zeros')
	if x.shape:
		len_x = x.shape[0]
	else:
		len_x = 1
	if len(coeff)==4:
		return np.exp(np.sum(coeff*np.column_stack((np.log(x)**3, np.log(x)**2, np.log(x), np.ones(len_x))),1))
	elif len(coeff)==2:
		return np.exp(np.sum(coeff*np.column_stack((np.log(x), np.ones(len_x))),1))
	else:
		raise Exception('length of coeff is incorrect')


# run the differential expression analysis
def run_glee():
	startTime = datetime.now()
	
	global filename_value, nA_value, nB_value, p_value, fbL_value, fbH_value, binchoice_value, fitType_value, fQ_value, niter_value, outID_value
	global progress_value
	progress_value.set('Processing...')

	# check the GUI options to make sure they are valid
	OE=[]
	if not filename_value.get().strip():
		OE.append('file_name: cannot be empty')
	else:
		try:
			open(filename_value.get())
		except:
			OE.append('Cannot open file: ' + filename_value.get())
	try:
		if nA_value.get()<=0:
			OE.append('num_replicates_A: must be positive')
	except Exception as nA_err:
		OE.append('num_replicates_A: %s' % nA_err)
	try:
		if nB_value.get()<=0:
			OE.append('num_replicates_B: must be positive')
	except Exception as nB_err:
		OE.append('num_replicates_B: %s' % nB_err)
	try:
		if p_value.get()<10 or p_value.get()>100:
			OE.append('num_bins: specify a value between 10 and 100, recommended value = 20')
	except Exception as p_err:
		OE.append('num_bins: %s' % p_err)
	try:
		if fbL_value.get()<0 or fbL_value.get()>10:
			OE.append('merge_low: specify a value between 0 and 10, recommended value = 1')
	except Exception as fbL_err:
		OE.append('merge_low: %s' % fbL_err)
	try:
		if fbH_value.get()<0 or fbH_value.get()>10:
			OE.append('merge_high: specify a value between 0 and 10, recommended value = 1')
	except Exception as fbH_err:
		OE.append('merge_high: %s' % fbH_err)
	try:
		if fQ_value.get()<=0 or fQ_value.get()>=1:
			OE.append('fit_quantile: specify a floating-point value between 0 and 1, recommended value = 0.5')
	except Exception as fQ_err:
		OE.append('fit_quantile: %s' % fQ_err)
	try:
		if niter_value.get()<1000 or niter_value.get()>50000:
			OE.append('num_iterations: specify a value between 1000 and 50000, recommended value = 10000')
	except Exception as niter_err:
		OE.append('num_iterations: %s' % niter_err)

	if OE:
		OE.insert(0,'Please correct the following errors')
		progress_value.set('')
		showerror('ERROR','%s' % '\n '.join(OE))
		return

	# the main processing begins here
	try:
		(filename,nA,nB,p,fbL,fbH,binchoice,fit_type,q,iter,out_id) = (filename_value.get(), nA_value.get(), nB_value.get(), p_value.get(), fbL_value.get(), fbH_value.get(), binchoice_value.get(), fitType_value.get(), fQ_value.get(), niter_value.get(), outID_value.get())
		
		LOG = out_id + '.log.txt'
		flog = open(LOG,'w')
		
		Prot,A,B = xlsread(filename,nA,nB)
		conditions = ('A','B')

		# Data sanity check: A,B should not contain only finite positive values
		if not np.all(np.isfinite(A)):
			raise Exception('A contains non-finite values')
		if np.any(A<0):
			raise Exception('A contains negative values')
		if not np.all(np.isfinite(B)):
			raise Exception('B contains non-finite values')
		if np.any(B<0):
			raise Exception('B contains negative values')
		# A and B should have the same number of rows
		if A.shape[0]!=B.shape[0]:
			raise Exception('A and B have unequal number of rows')

		# Calculate stats to analyze in each condition
		xbar = np.column_stack((A.mean(1),B.mean(1)))
		stdev = np.column_stack((A.std(axis=1,ddof=1),B.std(axis=1,ddof=1)))
		
		if nobinning_value.get():
			adjRsq = []; C = [];
			for j,cond in enumerate(conditions):
				flog.write('\n-- processing condition %s --\n' % cond)
				# retain non-zero values of xbar (and corresponding values of stdev)
				Fnz = xbar[:,j]!=0
				xbar_values = np.log(xbar[Fnz,j]); stdev_values = np.log(stdev[Fnz,j]);
				# remove non-finite values of stdev
				Ff = np.isfinite(stdev_values)
				xbar_values = xbar_values[Ff]; stdev_values = stdev_values[Ff];
				I=np.argsort(xbar_values); sorted_xbar_values = xbar_values[I]; sorted_stdev_values = stdev_values[I];
				# perform the fit
				X = sorted_xbar_values; Y = sorted_stdev_values;
				if fit_type.strip()=='linear':
					[coefs,r2,adjr2] = regress(X,Y,1)
					flog.write("adjRsq = %.6f\n" % adjr2)
					adjRsq.append(adjr2)
					C.append(coefs)
					Xhat = np.linspace(X[0],X[-1],20)
					Yhat = np.sum(coefs*np.column_stack((Xhat,np.ones(len(Xhat)))),1)
				elif fit_type.strip()=='cubic':
					[coefs,r2,adjr2] = regress(X,Y,3)
					flog.write("adjRsq = %.6f\n" % adjr2)
					adjRsq.append(adjr2)
					C.append(coefs)
					Xhat = np.linspace(X[0],X[-1],20)
					Yhat = np.sum(coefs*np.column_stack((Xhat**3,Xhat**2,Xhat,np.ones(len(Xhat)))),1)
				# leave out the outliers from the plot
				showpoints = (stdev_values>(np.mean(stdev_values)-5*np.std(stdev_values))) & (stdev_values<(np.mean(stdev_values)+5*np.std(stdev_values)))
				fig=plt.figure()
				ax=fig.add_subplot(111)
				ax.plot(xbar_values[showpoints],stdev_values[showpoints],'b.',Xhat,Yhat,'r-')
				ax.legend(('raw','fit (adjRsq='+str(adjRsq[j])+')'),'upper right')
				ax.grid(False)
				ax.set_xlabel('log(xbar)'); ax.set_ylabel('log(stdev)'); ax.set_title('Condition '+cond)
				fig.savefig(out_id+'.condition-'+cond+'.png',format='png')
				plt.close(fig)
				fig=plt.figure()
				ax=fig.add_subplot(111)
				ax.plot(1+np.arange(len(sorted_xbar_values)),sorted_xbar_values,'b.')
				ax.grid(False)
				ax.set_xlabel('protein #'); ax.set_ylabel('signal level (xbar)'); ax.set_title('Condition '+cond)
				fig.savefig(out_id+'.condition-'+cond+'.siglevel.png',format='png')
				plt.close(fig)
		else:
			OFILE = out_id + '.selected_points.txt'
			fout = open(OFILE,'w')
			fout.write('log(xbar)\tlog(stdev)\n')
			
			if binchoice.strip()=='equal':
				# ---- EQUAL SIZED BINS ----
				adjRsq = []; C = [];
				for j,cond in enumerate(conditions):
					flog.write('\n-- processing condition %s --\n' % cond)
					# retain non-zero values of xbar (and corresponding values of stdev)
					Fnz = xbar[:,j]!=0
					xbar_values = np.log(xbar[Fnz,j]); stdev_values = np.log(stdev[Fnz,j]);
					# remove non-finite values of stdev
					Ff = np.isfinite(stdev_values)
					xbar_values = xbar_values[Ff]; stdev_values = stdev_values[Ff];
					I=np.argsort(xbar_values); sorted_xbar_values = xbar_values[I]; sorted_stdev_values = stdev_values[I];

					# divide into p bins of equal size, take specified quantile in each bin
					L=np.floor(len(sorted_xbar_values)/p)
					X=np.empty(p); Y=np.empty(p);
					for i in range(p-1):
						X[i] = mquantiles(sorted_xbar_values[i*L:(i+1)*L],prob=q,alphap=0.5,betap=0.5)
						Y[i] = mquantiles(sorted_stdev_values[i*L:(i+1)*L],prob=q,alphap=0.5,betap=0.5)
					X[p-1] = mquantiles(sorted_xbar_values[(p-1)*L:],prob=q,alphap=0.5,betap=0.5)
					Y[p-1] = mquantiles(sorted_stdev_values[(p-1)*L:],prob=q,alphap=0.5,betap=0.5)
					I = np.isfinite(X) & np.isfinite(Y) # remove nan's if any
					X=X[I]; Y=Y[I];
					# print out selected points
					fout.write('---- condition %s ----\n' % cond)
					for i in range(len(X)):
						fout.write('%.6f\t%.6f\n' %(X[i],Y[i]))
					# perform the fit
					if fit_type.strip()=='linear':
						[coefs,r2,adjr2] = regress(X,Y,1)
						flog.write("adjRsq = %.6f\n" % adjr2)
						adjRsq.append(adjr2)
						C.append(coefs)
						Yhat = np.sum(coefs*np.column_stack((X,np.ones(len(X)))),1)
					elif fit_type.strip()=='cubic':
						[coefs,r2,adjr2] = regress(X,Y,3)
						flog.write("adjRsq = %.6f\n" % adjr2)
						adjRsq.append(adjr2)
						C.append(coefs)
						Yhat = np.sum(coefs*np.column_stack((X**3,X**2,X,np.ones(len(X)))),1)
					# leave out the outliers from the plot
					showpoints = (stdev_values>(np.mean(stdev_values)-5*np.std(stdev_values))) & (stdev_values<(np.mean(stdev_values)+5*np.std(stdev_values)))
					fig=plt.figure()
					ax=fig.add_subplot(111)
					ax.plot(xbar_values[showpoints],stdev_values[showpoints],'b.',X,Yhat,'r-')
					ax.legend(('raw','fit (adjRsq='+str(adjRsq[j])+')'),'upper right')
					ax.grid(False)
					ax.set_xlabel('log(xbar)'); ax.set_ylabel('log(stdev)'); ax.set_title('Condition '+cond)
					fig.savefig(out_id+'.condition-'+cond+'.png',format='png')
					plt.close(fig)
					fig=plt.figure()
					ax=fig.add_subplot(111)
					ax.plot(1+np.arange(len(sorted_xbar_values)),sorted_xbar_values,'b.')
					ax.grid(False)
					ax.set_xlabel('protein #'); ax.set_ylabel('signal level (xbar)'); ax.set_title('Condition '+cond)
					fig.savefig(out_id+'.condition-'+cond+'.siglevel.png',format='png')
					plt.close(fig)
			elif binchoice.strip()=='adaptive':
				adjRsq = []; C = [];
				for j,cond in enumerate(conditions):
					flog.write('\n-- processing condition %s --\n' % cond)
					# retain non-zero values of xbar (and corresponding values of stdev)
					Fnz = xbar[:,j]!=0
					xbar_values = np.log(xbar[Fnz,j]); stdev_values = np.log(stdev[Fnz,j]);
					# remove non-finite values of stdev
					Ff = np.isfinite(stdev_values)
					xbar_values = xbar_values[Ff]; stdev_values = stdev_values[Ff];
					I=np.argsort(xbar_values); sorted_xbar_values = xbar_values[I]; sorted_stdev_values = stdev_values[I];
					
					# divide the range of signal into p equal portions
					dx = (sorted_xbar_values[-1] - sorted_xbar_values[0])/p
					# Start and Stop will contain 0-based indexes
					Start=[]; Stop=[];
					for i in range(p-1):
						if not Start:
							start=0
						else:
							start=Stop[-1]+1
						stop = np.max(np.where(sorted_xbar_values<=(sorted_xbar_values[0]+(i+1)*dx)))
						if stop>=start:
							Start.append(start); Stop.append(stop)
					start=Stop[-1]+1; stop=len(sorted_xbar_values)-1
					if stop>=start:
						Start.append(start); Stop.append(stop)
					if len(Start)!=len(Stop):
						raise Exception('Start and Stop are not of equal length')
					# convert Start and Stop to numpy arrays to enable use of np.where()
					Start = np.array(Start); Stop = np.array(Stop)
					
					# merge the low-signal bins so that the lowest one contains atleast fbL% of the proteins
					if fbL>0:
						ind = np.min( np.where( Stop > ((fbL/100)*len(sorted_xbar_values)) ) )
						Start = np.append(np.array(0),Start[ind+1:]); Stop=Stop[ind:]
						flog.write('Percentage of proteins in the lowest bin = %2.6f\n' % (100*((Stop[0]-Start[0]+1)/len(sorted_xbar_values))))

					# merge the high-signal bins so that the highest one contains atleast fbH% of the proteins
					if fbH>0:
						ind = np.max( np.where( Stop < ( (1-(fbH/100))*len(sorted_xbar_values) ) ) )
						Start = Start[0:(ind+2)]; Stop = np.append(Stop[0:(ind+1)],Stop[-1])
						flog.write('Percentage of proteins in the highest bin = %2.6f\n' % (100*((Stop[-1]-Start[-1]+1)/len(sorted_xbar_values))))

					X=np.empty(len(Start)); Y=np.empty(len(Start));
					for i in range(len(X)):
						X[i] = mquantiles(sorted_xbar_values[Start[i]:Stop[i]+1],prob=q,alphap=0.5,betap=0.5)
						Y[i] = mquantiles(sorted_stdev_values[Start[i]:Stop[i]+1],prob=q,alphap=0.5,betap=0.5)
					
					I = np.isfinite(X) & np.isfinite(Y) # remove nan values if any
					X=X[I]; Y=Y[I];
					# print out selected points
					fout.write('---- condition %s ----\n' % cond)
					for i in range(len(X)):
						fout.write('%.6f\t%.6f\n' %(X[i],Y[i]))
					# perform the fit
					if fit_type.strip()=='linear':
						[coefs,r2,adjr2] = regress(X,Y,1)
						flog.write("adjRsq = %.6f\n" % adjr2)
						adjRsq.append(adjr2)
						C.append(coefs)
						Yhat = np.sum(coefs*np.column_stack((X,np.ones(len(X)))),1)
					elif fit_type.strip()=='cubic':
						[coefs,r2,adjr2] = regress(X,Y,3)
						flog.write("adjRsq = %.6f\n" % adjr2)
						adjRsq.append(adjr2)
						C.append(coefs)
						Yhat = np.sum(coefs*np.column_stack((X**3,X**2,X,np.ones(len(X)))),1)
					# leave out the outliers from the plot
					showpoints = (stdev_values>(np.mean(stdev_values)-5*np.std(stdev_values))) & (stdev_values<(np.mean(stdev_values)+5*np.std(stdev_values)))
					fig=plt.figure()
					ax=fig.add_subplot(111)
					ax.plot(xbar_values[showpoints],stdev_values[showpoints],'b.',X,Yhat,'r-')
					ax.legend(('raw','fit (adjRsq='+str(adjRsq[j])+')'),'upper right')
					ax.grid(False)
					ax.set_xlabel('log(xbar)'); ax.set_ylabel('log(stdev)'); ax.set_title('Condition '+cond)
					fig.savefig(out_id+'.condition-'+cond+'.png',format='png')
					plt.close(fig)
					fig=plt.figure()
					ax=fig.add_subplot(111)
					ax.plot(1+np.arange(len(sorted_xbar_values)),sorted_xbar_values,'b.')
					ax.grid(False)
					ax.set_xlabel('protein #'); ax.set_ylabel('signal level (xbar)'); ax.set_title('Condition '+cond)
					fig.savefig(out_id+'.condition-'+cond+'.siglevel.png',format='png')
					plt.close(fig)
			fout.close()

		# choose the fit that gives better adjRsq
		bestfitind = np.argsort(adjRsq)[1]
		coeff = C[bestfitind]

		# replace xbar zeroes with min-positive xbar from the same condition
		for j in range(xbar.shape[1]):
			xbar[xbar[:,j]<=0,j] = np.min(xbar[xbar[:,j]>0,j])
		min_xbar_value=np.min(xbar,0)

		# calculate model-based STN
		model_stn = (xbar[:,1]-xbar[:,0])/(calcs(xbar[:,0],coeff)+calcs(xbar[:,1],coeff))
		if not np.all(np.isfinite(model_stn)):
			raise Exception('model_stn contains non-finite values')

		# calculate null distribution of model_stn using the specified baseline condition
		baselinecol=bestfitind # condition that is to be used as baseline
		flog.write('\nUsing condition %s as baseline since its adjRsq is higher\n' % conditions[baselinecol])
		# notation followed: A = baseline, B = other
		conditions_order=''
		if conditions[baselinecol]=='B':
			# interchange A and B so A stays the baseline
			A,B = B,A
			nA,nB = nB,nA
			conditions_order = 'reversed'
		model_stn_dist_len = iter*A.shape[0]
		model_stn_dist = np.empty((model_stn_dist_len))
		model_stn_dist.fill(np.nan)
		flog.write('\n-- doing resampling --\n')
		for i in range(A.shape[0]):
			if np.remainder(i,100)==0:
				flog.write('processing %d of %d...\n' % (i,A.shape[0]))
			a = A[i,:]
			# sample values with replacement
			Astar = a[np.random.randint(nA,size=(iter,nA))]; Bstar = a[np.random.randint(nB,size=(iter,nB))];
			xbar_Astar=Astar.mean(1); xbar_Bstar=Bstar.mean(1); 
			xbar_Astar[xbar_Astar==0] = min_xbar_value[baselinecol]
			xbar_Bstar[xbar_Bstar==0] = min_xbar_value[baselinecol]
			this_dist = (xbar_Bstar - xbar_Astar)/(calcs(xbar_Astar,coeff) + calcs(xbar_Bstar,coeff))
			if not isinstance(this_dist,np.ndarray):
				raise Exception('this_dist is not the right type')
			if this_dist.shape[0]!=iter:
				raise Exception('this_dist is not right size')
			if not np.all(np.isfinite(this_dist)):
				raise Exception('check resampled distribution for protein i=%d' % i)
			model_stn_dist[(i*iter):((i+1)*iter)] = this_dist

		if not np.all(np.isfinite(model_stn_dist)):
			raise Exception('model_stn_dist contains non-finite values')

		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.hist(model_stn_dist,bins=100)
		ax.set_title('Histogram of STN distribution')
		fig.savefig(out_id+'.stn_distr.png',format='png')
		plt.close(fig)

		# calculate p-values
		pValue=np.empty(A.shape[0])
		pValue.fill(np.nan)
		for i in range(A.shape[0]):
			c = np.sum(model_stn_dist > model_stn[i]) / model_stn_dist_len
			pValue[i] = 2*np.min((c,1-c))

		# plot STN vs p-values
		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.plot(model_stn,pValue,'r*')
		ax.set_xlabel('STN'); ax.set_ylabel('p-value');
		fig.savefig(out_id+'.stn_pvalue.png',format='png')
		plt.close(fig)

		# sort in increasing order of p-value
		order=np.argsort(pValue)

		# print out DEG
		if conditions_order.strip():
			# un-reverse A,B so they get back into the originally specified order
			A,B = B,A
			nA,nB = nB,nA
		OFILE = out_id + '.DEG.txt'
		fout = open(OFILE,'w')
		fout.write('Protein\t')
		for j in range(nA):
			fout.write('A_'+str(j+1)+'\t')
		for j in range(nB):
			fout.write('B_'+str(j+1)+'\t')
		fout.write('xbar(A)\ts(A)\txbar(B)\ts(B)\tSTN\tpValue\n')
		for k in order:
			fout.write('%s' % Prot[k])
			for j in range(nA):
				fout.write('\t'+str(A[k,j]))
			for j in range(nB):
				fout.write('\t'+str(B[k,j]))
			for value in (xbar[k,0],calcs(xbar[k,0],coeff),xbar[k,1],calcs(xbar[k,1],coeff),model_stn[k],pValue[k]):
				fout.write('\t%6.12f' % value)
			fout.write('\n')
		fout.close()
		
		stopTime = datetime.now()
		flog.write('\nTime taken for the analysis (in minutes) = %3.2f\n' % ((stopTime-startTime).total_seconds()/60))
		flog.close()
		progress_value.set('DONE! Output files written to\n' + getcwd())
	except Exception as err:
		progress_value.set('')
		showerror('ERROR','%s' % err)


# reset gui fields to their default values
def reset_fields():
	global filename_value, nA_value, nB_value, p_value, fbL_value, fbH_value, equal, cubic, fQ_value, niter_value, outID_value
	filename_value.set(filename_default); nA_value.set(nA_default); nB_value.set(nB_default); p_value.set(p_default); fbL_value.set(fbL_default); fbH_value.set(fbH_default); 
	equal.select(); cubic.select(); fQ_value.set(fQ_default); niter_value.set(niter_default); outID_value.set(outID_default)
	global progress_value
	progress_value.set('')

# check the status of binning choice and disable unnecessary options accordingly
def check_binning():
	global nobinning_value, p_entry, fbL_entry, fbH_entry, equal, adaptive, fQ_entry
	if nobinning_value.get():
		p_entry.config(state='disabled')
		fbL_entry.config(state='disabled')
		fbH_entry.config(state='disabled')
		equal.config(state='disabled')
		adaptive.config(state='disabled')
		fQ_entry.config(state='disabled')
	else:
		p_entry.config(state='normal')
		fbL_entry.config(state='normal')
		fbH_entry.config(state='normal')
		equal.config(state='normal')
		adaptive.config(state='normal')
		fQ_entry.config(state='normal')
		check_binchoice()

# check if adaptive binning is selected and enable merging
def check_binchoice():
	global binchoice_value, fbL_entry, fbH_entry
	if binchoice_value.get().strip()=='adaptive':
		fbL_entry.config(state='normal')
		fbH_entry.config(state='normal')
	else:
		fbL_entry.config(state='disabled')
		fbH_entry.config(state='disabled')

# Start preparing the GUI
root = Tk()
root.geometry('300x500')
root.resizable(width=0,height=0)

# -- create all the widgets --
names = ('file_name','num_replicates_A','num_replicates_B','num_bins','merge_low','merge_high','bin_choice','fit_type','fit_quantile','num_iterations','output_id')
head_label = Label(root, text='GLEE : differential protein expression analysis')
filename_value = StringVar()
filename_button = Button(root, text=names[0], command = lambda : filename_value.set(askopenfilename()))
filename_entry = Entry(root, textvariable=filename_value, justify='left')
nA_label = Label(root, text=names[1])
nA_value = IntVar(); nA_value.set(nA_default)
nA_entry = Entry(root, textvariable=nA_value)
nB_label = Label(root, text=names[2])
nB_value = IntVar(); nB_value.set(nB_default)
nB_entry = Entry(root, textvariable=nB_value)
nobinning_label = Label(root, text='DO NOT USE BINNING')
nobinning_value = IntVar(); nobinning_value.set(1)
nobinning_check = Checkbutton(root, variable=nobinning_value, onvalue=1, offvalue=0, command=check_binning)
p_label = Label(root, text=names[3])
p_value = IntVar(); p_value.set(p_default)
p_entry = Entry(root, textvariable=p_value)
fbL_label = Label(root, text=names[4])
fbL_value = DoubleVar(); fbL_value.set(fbL_default)
fbL_entry = Entry(root, textvariable=fbL_value)
fbH_label = Label(root, text=names[5])
fbH_value = DoubleVar(); fbH_value.set(fbH_default)
fbH_entry = Entry(root, textvariable=fbH_value)
binchoice_label = Label(root, text=names[6])
binchoice_frame = Frame(root)
binchoice_value = StringVar()
equal = Radiobutton(binchoice_frame,text='equal',variable=binchoice_value,value='equal',command=check_binchoice)
equal.select()
adaptive = Radiobutton(binchoice_frame,text='adaptive',variable=binchoice_value,value='adaptive',command=check_binchoice)
equal.grid(row=0,column=0)
adaptive.grid(row=0,column=1)
binchoice_frame.grid()
fitType_label = Label(root, text=names[7])
fitType_frame = Frame(root)
fitType_value = StringVar()
linear = Radiobutton(fitType_frame,text='linear',variable=fitType_value,value='linear')
cubic = Radiobutton(fitType_frame,text='cubic',variable=fitType_value,value='cubic')
cubic.select()
linear.grid(row=0,column=0)
cubic.grid(row=0,column=1)
fitType_frame.grid()
fQ_label = Label(root, text=names[8])
fQ_value = DoubleVar(); fQ_value.set(fQ_default)
fQ_entry = Entry(root, textvariable=fQ_value)
niter_label = Label(root, text=names[9])
niter_value = IntVar(); niter_value.set(niter_default)
niter_entry = Entry(root, textvariable=niter_value)
outID_label = Label(root, text=names[10])
outID_value = StringVar(); outID_value.set(outID_default)
outID_entry = Entry(root, textvariable=outID_value)
submit_button = Button(root, text='RUN', command=run_glee)
progress_value=StringVar(); progress_value.set(' \n ')
progress_label = Label(root, textvariable=progress_value)
reset_button = Button(root, text='RESET', command = reset_fields)
help_button = Button(root, text='HELP', command = lambda : open_new('http://sites.google.com/site/lalitp/glee'))
check_binning()
check_binchoice()

# -- layout the widgets --
head_label.grid(row=0,column=0, padx=5,pady=5, columnspan=2)
filename_button.grid(row=1,column=0, padx=5,pady=5, sticky='nsew')
filename_entry.grid(row=1, column=1, padx=5,pady=5)
nA_label.grid(row=2,column=0, padx=5,pady=5)
nA_entry.grid(row=2, column=1, padx=5,pady=5)
nB_label.grid(row=3,column=0, padx=5,pady=5)
nB_entry.grid(row=3, column=1, padx=5,pady=5)
fitType_label.grid(row=4,column=0, padx=5,pady=5)
fitType_frame.grid(row=4,column=1, padx=5,pady=5)
niter_label.grid(row=5,column=0, padx=5,pady=5)
niter_entry.grid(row=5,column=1, padx=5,pady=5)
nobinning_label.grid(row=6,column=0)
nobinning_check.grid(row=6,column=1)
p_label.grid(row=7,column=0, padx=5,pady=5)
p_entry.grid(row=7,column=1, padx=5,pady=5)
fbL_label.grid(row=8,column=0, padx=5,pady=5)
fbL_entry.grid(row=8,column=1, padx=5,pady=5)
fbH_label.grid(row=9,column=0, padx=5,pady=5)
fbH_entry.grid(row=9,column=1, padx=5,pady=5)
binchoice_label.grid(row=10,column=0, padx=5,pady=5)
binchoice_frame.grid(row=10,column=1, padx=5,pady=5)
fQ_label.grid(row=11,column=0, padx=5,pady=5)
fQ_entry.grid(row=11,column=1, padx=5,pady=5)
outID_label.grid(row=12,column=0, padx=5,pady=5)
outID_entry.grid(row=12,column=1, padx=5,pady=5)
submit_button.grid(row=14,column=0, padx=5,pady=5, columnspan=2, sticky='nsew')
progress_label.grid(row=15,column=0, padx=5,pady=5, columnspan=2)
reset_button.grid(row=17,column=0, padx=5,pady=5, sticky='nsew')
help_button.grid(row=17,column=1, padx=5,pady=5, sticky='nsew')
root.grid()

# -- set scaling and run --
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.mainloop()

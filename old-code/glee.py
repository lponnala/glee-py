from __future__ import division
import re
import xlrd
import sys
import numpy as np
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt

# ---- COMMENTS ----
# [because we don't care about which sample is greater] Why 2*min(c,1-c) for p-value? I think min(c,1-c) is enough...
# Print out the name of the sample being used as baseline (e.g. WT or clpS) instead of A or B
# [checked to be same order as specified: A first, B next] If we end up interchanging A and B, then in which order are they written to the final output file? What we print out under A there - is it the original A or the "bestfit" A?
# [doesn't help] Import only the parts needed from the external modules such as numpy, xlrd, sys, etc... might reduce the output .exe size
# [done] validate the input file and options upon submit and print errors to a dialog box
# [doesn't appear though] show "Processing..." in the progress label from the time validation occurs until final results are written
# [a big one added around run_glee()] try-catch statements must be added at each step ensuring that all errors are piped to dialog boxes
# when merging bins, make sure there are enough left over (in cases where there's just a few proteins to compare)
# [done] use enumerate instead of conditions[j]
# [done in gui.py] merge bins only if fbL!=0 (fbH!=0)
# -------------------

# function to read the excel spreadsheet
def xlsread(filename,nA,nB):
	wb=xlrd.open_workbook(filename)
	sh=wb.sheet_by_index(0)
	# check if the number of columns are correct
	if sh.ncols != 1+nA+nB:
		sys.exit('incorrect number of columns in spreadsheet...')
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
		print "x and y are of unequal length! Quitting..."; sys.exit()
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
		print 'x contains zeros!'; sys.exit()
	if x.shape:
		len_x = x.shape[0]
	else:
		len_x = 1
	if len(coeff)==4:
		return np.exp(np.sum(coeff*np.column_stack((np.log(x)**3, np.log(x)**2, np.log(x), np.ones(len_x))),1))
	elif len(coeff)==2:
		return np.exp(np.sum(coeff*np.column_stack((np.log(x), np.ones(len_x))),1))
	else:
		print 'length of coeff is incorrect!'; sys.exit()

# false discovery rate (fdr)
def fdr(P,q):
	M=len(P)
	I=np.argsort(P)
	Ps=P[I]
	Is=q*((1+np.arange(M,dtype='float'))/M)
	H=np.zeros(M)
	if np.any(Ps<=Is):
		i_q=np.max(np.where(Ps<=Is)[0])
		H[I[0:i_q+1]]=1
	return H

optsfile=raw_input('Enter the options file name: ')
opts={}
try:
	with open(optsfile) as file:
		for line in file:
			m=re.findall('(\S+)\s*=\s*(\S+)',line)
			if m:
				opts[m[0][0]]=m[0][1]
except IOError as e:
	print 'Error: Could not find options file!'

OE=[] # catch options errors
for key in opts:
	# file_name : string, make sure file can be read
	if key=='file_name':
		try:
			with open(opts[key]) as file:
				filename=opts[key]
		except IOError as e:
			print 'Error: Could not find file: ' + opts[key]
	# num_replicates_A / mum_replicates_B : integer
	if key=='num_replicates_A':
		nA = int(opts[key]);
	if key=='num_replicates_B':
		nB = int(opts[key]);
	# num_bins : integer in [10 100]
	if key=='num_bins':
		p = int(opts[key]);
		if p<10 or p>100:
			OE.append('specify a value between 10 and 100 for ' + key + ', recommended value = 20')
	# num_iterations : integer in [100 10000]
	if key=='num_iterations':
		iter = int(opts[key])
		if iter<100 or iter>10000:
			OE.append('specify a value between 100 and 10000 for ' + key + ', recommended value = 1000')
	# merge_low / merge_high : float
	if key=='merge_low':
		fbL = float(opts[key])
		if fbL<0 or fbL>10:
			OE.append('specify a value between 0 and 10 for ' + key + ', recommended value = 1')
		# fbH = float(opts[key])
	if key=='merge_high':
		fbH = float(opts[key])
		if fbH<0 or fbH>10:
			OE.append('specify a value between 0 and 10 for ' + key + ', recommended value = 1')
	# fit_quantile : float in (0,1)
	if key=='fit_quantile':
		q=float(opts[key])
		if q<=0 or q>=1:
			OE.append('specify a floating-point value for ' + key + ', recommended value = 0.5')
	# fdr_level : float in (0,1)
	if key=='fdr_level':
		fdr_level=float(opts[key])
		if fdr_level<=0 or fdr_level>=1:
			OE.append('specify a value between 0 and 1 for ' + key + ', recommended value =0.05')
	# bin_choice / fit_type / output_id : string
	if key=='bin_choice':
		binchoice = opts[key].lower()
	if key=='fit_type':
		fit_type = opts[key].lower()
	if key=='output_id':
		out_id = opts[key]

if OE:
	print '\nErrors found in the options file!'
	print '\n'.join(OE)
	print 'Please correct the above errors and re-run'
	sys.exit()

Prot,A,B = xlsread(filename,nA,nB)
conditions = ('A','B')

# DATA SANITY CHECK
# A,B should not contain only finite positive values
if not np.all(np.isfinite(A)):
	print 'A contains non-finite values... Quitting\n'; sys.exit()
if np.any(A<0):
	print 'A contains negative values... Quitting\n'; sys.exit()
if not np.all(np.isfinite(B)):
	print 'B contains non-finite values... Quitting\n'; sys.exit()
if np.any(B<0):
	print 'B contains negative values... Quitting\n'; sys.exit()
# A and B should have the same number of rows
if A.shape[0]!=B.shape[0]:
	print 'A and B have unequal number of rows... Quitting\n'; sys.exit()

# START THE ANALYSIS
xbar = np.column_stack((A.mean(1),B.mean(1)))
stdev = np.column_stack((A.std(axis=1,ddof=1),B.std(axis=1,ddof=1)))

# fo=open('stdev.txt','w')
# for i in range(stdev.shape[0]):
	# fo.write('%.6f\t%.6f\n' %(stdev[i,0],stdev[i,1]))
# fo.close()

OFILE = out_id + '.selected_points.txt'
fout = open(OFILE,'w')
fout.write('log(xbar)\tlog(stdev)\n')

if binchoice.strip()=='equal':
	# ---- EQUAL SIZED BINS ----
	adjRsq = []; C = [];
	for j,cond in enumerate(conditions): #range(len(conditions)):
		print '-- processing condition %s --\n' % cond
		# retain non-zero values of xbar (and corresponding values of stdev)
		Fnz = xbar[:,j]!=0
		xbar_values = np.log(xbar[Fnz,j]); stdev_values = np.log(stdev[Fnz,j]);
		# remove non-finite values of stdev
		Ff = np.isfinite(stdev_values)
		xbar_values = xbar_values[Ff]; stdev_values = stdev_values[Ff];
		I=np.argsort(xbar_values); sorted_xbar_values = xbar_values[I]; sorted_stdev_values = stdev_values[I];
		
		# fo=open('sorted.txt','w')
		# for i in range(len(sorted_xbar_values)):
			# fo.write('%.6f\t%.6f\n' % (sorted_xbar_values[i],sorted_stdev_values[i]))
		# fo.close()
		
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
			print "adjRsq = %.6f\n" % adjr2
			adjRsq.append(adjr2)
			C.append(coefs)
			Yhat = np.sum(coefs*np.column_stack((X,np.ones(len(X)))),1)
		elif fit_type.strip()=='cubic':
			[coefs,r2,adjr2] = regress(X,Y,3)
			print "adjRsq = %.6f\n" % adjr2
			adjRsq.append(adjr2)
			C.append(coefs)
			Yhat = np.sum(coefs*np.column_stack((X**3,X**2,X,np.ones(len(X)))),1)
		else:
			error('Could not understand fit_type\n')
		# leave out the outliers from the plot
		showpoints = (stdev_values>(np.mean(stdev_values)-5*np.std(stdev_values))) & (stdev_values<(np.mean(stdev_values)+5*np.std(stdev_values)))
		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.plot(xbar_values[showpoints],stdev_values[showpoints],'b.',X,Yhat,'r-')
		ax.legend(('raw','fit (adjRsq='+str(adjRsq[j])+')'),'upper right')
		ax.grid(False)
		ax.set_xlabel('log(xbar)'); ax.set_ylabel('log(stdev)'); ax.set_title('SAMPLE-'+str(j))
		fig.savefig(out_id+'.sample-'+str(j+1)+'.png',format='png')
		plt.close(fig)
		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.plot(1+np.arange(len(sorted_xbar_values)),sorted_xbar_values,'b.')
		ax.grid(False)
		ax.set_xlabel('protein #'); ax.set_ylabel('signal level (xbar)'); ax.set_title('SAMPLE-'+str(j))
		fig.savefig(out_id+'.sample-'+str(j+1)+'.siglevel.png',format='png')
		plt.close(fig)
elif binchoice.strip()=='adaptive':
	adjRsq = []; C = [];
	for j,cond in enumerate(conditions): # range(len(conditions)):
		print '-- processing condition %s --\n' % cond
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
			print 'Start and Stop are not of equal length'
			sys.exit()
		# convert Start and Stop to numpy arrays to enable use of np.where()
		Start = np.array(Start); Stop = np.array(Stop)
		
		# merge the low-signal bins so that the lowest one contains atleast fbL% of the proteins
		ind = np.min( np.where( Stop > ((fbL/100)*len(sorted_xbar_values)) ) )
		Start = np.append(np.array(0),Start[ind+1:]); Stop=Stop[ind:]
		print 'Percentage of proteins in the lowest bin = %2.6f' % (100*((Stop[0]-Start[0]+1)/len(sorted_xbar_values)))
		
		# merge the high-signal bins so that the highest one contains atleast fbH% of the proteins
		ind = np.max( np.where( Stop < ( (1-(fbH/100))*len(sorted_xbar_values) ) ) )
		Start = Start[0:(ind+2)]; Stop = np.append(Stop[0:(ind+1)],Stop[-1])
		print 'Percentage of proteins in the highest bin = %2.6f' % (100*((Stop[-1]-Start[-1]+1)/len(sorted_xbar_values)))
		
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
			print "adjRsq = %.6f\n" % adjr2
			adjRsq.append(adjr2)
			C.append(coefs)
			Yhat = np.sum(coefs*np.column_stack((X,np.ones(len(X)))),1)
		elif fit_type.strip()=='cubic':
			[coefs,r2,adjr2] = regress(X,Y,3)
			print "adjRsq = %.6f\n" % adjr2
			adjRsq.append(adjr2)
			C.append(coefs)
			Yhat = np.sum(coefs*np.column_stack((X**3,X**2,X,np.ones(len(X)))),1)
		else:
			error('Could not understand fit_type\n')
		# leave out the outliers from the plot
		showpoints = (stdev_values>(np.mean(stdev_values)-5*np.std(stdev_values))) & (stdev_values<(np.mean(stdev_values)+5*np.std(stdev_values)))
		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.plot(xbar_values[showpoints],stdev_values[showpoints],'b.',X,Yhat,'r-')
		ax.legend(('raw','fit (adjRsq='+str(adjRsq[j])+')'),'upper right')
		ax.grid(False)
		ax.set_xlabel('log(xbar)'); ax.set_ylabel('log(stdev)'); ax.set_title('SAMPLE-'+str(j))
		fig.savefig(out_id+'.sample-'+str(j+1)+'.png',format='png')
		plt.close(fig)
		fig=plt.figure()
		ax=fig.add_subplot(111)
		ax.plot(1+np.arange(len(sorted_xbar_values)),sorted_xbar_values,'b.')
		ax.grid(False)
		ax.set_xlabel('protein #'); ax.set_ylabel('signal level (xbar)'); ax.set_title('SAMPLE-'+str(j))
		fig.savefig(out_id+'.sample-'+str(j+1)+'.siglevel.png',format='png')
		plt.close(fig)

fout.close()

# choose the fit that gives better adjRsq
bestfitind = np.argsort(adjRsq)[1]
coeff = C[bestfitind]

# replace xbar zeroes with min-positive xbar from the same sample
for j in range(xbar.shape[1]):
	xbar[xbar[:,j]<=0,j] = np.min(xbar[xbar[:,j]>0,j])
min_xbar_value=np.min(xbar,0)

# calculate model-based STN
model_stn = (xbar[:,1]-xbar[:,0])/(calcs(xbar[:,0],coeff)+calcs(xbar[:,1],coeff))
if not np.all(np.isfinite(model_stn)):
	print 'Error: model_stn contains non-finite values...Quitting'; sys.exit()

# fo=open('model_stn.txt','w')
# for i in range(model_stn.shape[0]):
	# fo.write('%.6f\n' % model_stn[i])
# fo.close()

# fo=open('model_stn_dist.txt','w')

# calculate null distribution of model_stn using the specified baseline condition
baselinecol=bestfitind # sample that is to be used as baseline
print 'Using condition %s as baseline since its adjRsq is higher' % conditions[baselinecol]
# notation followed: A = baseline, B = other
if conditions[baselinecol]=='B':
	# interchange A and B so A stays the baseline
	A,B = B,A
	nA,nB = nB,nA
model_stn_dist = np.empty((iter*A.shape[0]))
model_stn_dist.fill(np.nan)
print '-- doing resampling --'
for i in range(A.shape[0]):
	if np.remainder(i,100)==0:
		print 'processing %d of %d...\n' % (i,A.shape[0])
	a = A[i,:]
	# sample values with replacement
	Astar = a[np.random.randint(nA,size=(iter,nA))]; Bstar = a[np.random.randint(nB,size=(iter,nB))];
	xbar_Astar=Astar.mean(1); xbar_Bstar=Bstar.mean(1); 
	xbar_Astar[xbar_Astar==0] = min_xbar_value[baselinecol]
	xbar_Bstar[xbar_Bstar==0] = min_xbar_value[baselinecol]
	this_dist = (xbar_Bstar - xbar_Astar)/(calcs(xbar_Astar,coeff) + calcs(xbar_Bstar,coeff))
	if not isinstance(this_dist,np.ndarray):
		sys.exit('this_dist is not the right type')
	if this_dist.shape[0]!=iter:
		sys.exit('this_dist is not right size')
	if not np.all(np.isfinite(this_dist)):
		print 'check resampled distribution for protein i=', i
		sys.exit()
	# fo.write('-- i = %d --\n' % i)
	# for v in this_dist:
		# fo.write('%.6f\n' % v)
	model_stn_dist[(i*iter):((i+1)*iter)] = this_dist
# fo.close()

if not np.all(np.isfinite(model_stn_dist)):
	print 'model_stn_dist contains non-finite values'
	sys.exit()

fig=plt.figure()
ax=fig.add_subplot(111)
# (_,_,_)=ax.hist(model_stn_dist,bins=100,range=(),normed=True) #,bins=100)
ax.hist(model_stn_dist,bins=100)
ax.set_title('Histogram of stn distribution')
fig.savefig(out_id+'.stn_distr.png',format='png')
plt.close(fig)

# calculate p-values
pValue=np.empty(A.shape[0])
pValue.fill(np.nan)
for i in range(A.shape[0]):
	I = model_stn_dist > model_stn[i]
	if np.any(I):
		c = np.count_nonzero(I)/len(model_stn_dist)
		pValue[i] = 2*np.min((c,1-c))
	else:
		pValue[i] = 0

# plot STN vs p-values
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(model_stn,pValue,'r*')
ax.set_xlabel('STN'); ax.set_ylabel('p-value');
fig.savefig(out_id+'.stn_pvalue.png',format='png')
plt.close(fig)

# apply fdr cutoff
H=fdr(pValue,fdr_level)
print 'number of DEG detected = %d\n' % np.count_nonzero(H!=0)
order=np.argsort(pValue)

# pValue=pValue[order]; H=H[order]; model_stn=model_stn[order]; xbar=xbar[order,:];
# # Prot=Prot[order];
# Prots=[]
# for k in order:
	# Prots.append(Prot[k])
# A=A[order,:]
# B=B[order,:]

# print out DEG
OFILE = out_id + '.DEG.txt'
fout = open(OFILE,'w')
fout.write('Protein\t')
for j in range(nA):
	fout.write('A_'+str(j+1)+'\t')
for j in range(nB):
	fout.write('B_'+str(j+1)+'\t')
fout.write('xbar(A)\ts(A)\txbar(B)\ts(B)\tSTN\tpValue\tSignifDiffExp\n')
for k in order:
	fout.write('%s' % Prot[k])
	for j in range(nA):
		fout.write('\t'+str(A[k,j]))
	for j in range(nB):
		fout.write('\t'+str(B[k,j]))
	for value in (xbar[k,0],calcs(xbar[k,0],coeff),xbar[k,1],calcs(xbar[k,1],coeff),model_stn[k],pValue[k]):
		fout.write('\t%6.12f' % value)
	if H[k]==1:
		fout.write('\tYes\n')
	else:
		fout.write('\tNo\n')

# for i in range(len(H)):
	# fout.write('%s' % Prots[i])
	# for j in range(nA):
		# fout.write('\t'+str(A[i,j]))
	# for j in range(nB):
		# fout.write('\t'+str(B[i,j]))
	# for value in (xbar[i,1],calcs(xbar[i,1],coeff),xbar[i,2],calcs(xbar[i,2],coeff),model_stn[i],pValue[i]):
		# fout.write('\t%6.12f' % value)
	# if H[i]==1:
		# fout.write('\tYes\n')
	# else:
		# fout.write('\tNo\n')

fout.close()


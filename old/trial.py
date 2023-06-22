
# ------------------------------------------------------------
from __future__ import division
from Tkinter import Tk, StringVar, DoubleVar, IntVar, Label, Button, Entry, Frame, Radiobutton, Checkbutton

def run_glee():
	global nobinning_value
	if nobinning_value.get():
		print 'checked'
	else:
		print 'not checked'

root = Tk()

def check_binning():
	global nobinning_value, p_entry, equal, adaptive
	if nobinning_value.get():
		p_entry.config(state='disabled')
		equal.config(state='disabled')
		adaptive.config(state='disabled')		
	else:
		p_entry.config(state='normal')
		equal.config(state='normal')
		adaptive.config(state='normal')

# check if adaptive binning is selected and enable merging
def check_binchoice():
	global binchoice_value, fbL_entry, fbH_entry
	if binchoice_value.get().strip()=='adaptive':
		print 'adaptive'
	else:
		print 'equal'

nobinning_label = Label(root, text='No binning')
nobinning_value = IntVar(); nobinning_value.set(1)
nobinning_check = Checkbutton(root, variable=nobinning_value, onvalue=1, offvalue=0, command=check_binning)
p_label = Label(root, text='num_bins')
p_value = IntVar(); p_value.set(20)
p_entry = Entry(root, textvariable=p_value)
binchoice_label = Label(root, text='binchoice')
binchoice_frame = Frame(root)
binchoice_value = StringVar()
equal = Radiobutton(binchoice_frame,text='equal',variable=binchoice_value,value='equal')
equal.select()
adaptive = Radiobutton(binchoice_frame,text='adaptive',variable=binchoice_value,value='adaptive',command=check_binchoice)
equal.grid(row=0,column=0)
adaptive.grid(row=0,column=1)
binchoice_frame.grid()
check_binning()
check_binchoice()
submit_button = Button(root, text='RUN', command=run_glee)

nobinning_label.grid(row=0,column=0)
nobinning_check.grid(row=0,column=1)
p_label.grid(row=1,column=0, padx=5,pady=5)
p_entry.grid(row=1,column=1, padx=5,pady=5)
binchoice_label.grid(row=2,column=0, padx=5,pady=5)
binchoice_frame.grid(row=2,column=1, padx=5,pady=5)
submit_button.grid(row=3,column=0, padx=5,pady=5, columnspan=2, sticky='nsew')

root.grid()

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.mainloop()
# ------------------------------------------------------------


# reset -f
# from Tkinter import *
# root = Tk()
# cb_value = IntVar(); cb_value.set(1)
# cb = Checkbutton(root,variable=cb_value,onvalue=1,offvalue=0)


# from __future__ import division
# from Tkinter import Tk, StringVar, DoubleVar, IntVar, Label, Button, Entry, Frame, Radiobutton
# from tkMessageBox import showerror

# nA_default=0

# def run_glee():
	# '''
	# do nothing
	# '''
	# global nA_value
	# print "inside the function"
	# try:
		# # nA_value.get()
		# if nA_value.get()<=0:
			# print 'num_replicates_A: must be positive'
	# except Exception as err:
		# showerror('ERROR','%s' % err)
	# # if not isinstance(nA_value.get(),int):
		# # print "not a int"
	# # print type(nA_value.get())

# root = Tk()
# # root.geometry('300x500')
# # root.resizable(width=0,height=0)

# nA_label = Label(root, text='num_replicates_A')
# nA_value = IntVar(); nA_value.set(nA_default)
# nA_entry = Entry(root, textvariable=nA_value)

# # nA_label = Label(root, text='num_replicates_A')
# # nA_value = StringVar(); nA_value.set('A')
# # nA_entry = Entry(root, textvariable=nA_value)

# submit_button = Button(root, text='RUN', command=run_glee)

# nA_label.grid(row=3,column=0, padx=5,pady=5)
# nA_entry.grid(row=3, column=1, padx=5,pady=5)
# submit_button.grid(row=14,column=0, padx=5,pady=5, columnspan=2, sticky='nsew')
# root.grid()

# root.columnconfigure(0, weight=1)
# root.columnconfigure(1, weight=1)
# root.mainloop()



# from __future__ import division
# import numpy as np
# import random
# from datetime import datetime

# startTime = datetime.now()

# N=10000000
# model_stn_dist = np.random.random_sample((N))
# values = [0.2, 0.4, 0.6, 0.8]
# for v in values:
	# I = model_stn_dist > v
	# if np.any(I):
		# c = np.count_nonzero(I)/len(model_stn_dist)
		# print 'pvalue = ', 2*np.min((c,1-c))

# stopTime = datetime.now()
# print 'time taken = %s' % (stopTime-startTime)



# from Tkinter import Tk, Frame, Label, Button
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# # plt.ioff()

# def run_glee():
	# print 'called'
	# fig=plt.figure()
	# ax=fig.add_subplot(111)
	# ax.plot([1, 2, 3, 4, 5],[10,20,30,40,50],'b*')
	# fig.savefig('temp.png',format='png')
	# plt.close(fig)

# root = Tk()
# root.geometry('300x500')
# root.resizable(width=0,height=0)

# head_label = Label(root, text='GLEE : differential protein expression analysis')
# submit_button = Button(root, text='RUN', command=run_glee)

# head_label.grid(row=0,column=0, padx=5,pady=5, columnspan=2)
# submit_button.grid(row=14,column=0, padx=5,pady=5, columnspan=2, sticky='nsew')

# root.columnconfigure(0, weight=1)
# root.columnconfigure(1, weight=1)
# root.mainloop()



# filename='s-c.xls'
# if open(filename):
	# print 'can open'
# else:
	# print 'cannot open'


# import random
# import numpy as np

# N=300; K=6
# pv=np.empty((N,K))
# for i in range(N):
	# for j in range(K):
		# pv[i,j]=random.uniform(0,0.01)
# np.savetxt('randvals.txt',pv,fmt='%.8f',delimiter='\t',newline='\n')



# try:
	# x=10/0
# except Exception as err:
	# print '%s' % err


# # print the options set via the GUI to a file
# def print_options(Names,Values,OFILE):
	# fo=open(OFILE,'w')
	# for (n,v) in zip(Names,Values):
		# fo.write(n + ' : ' + str(v) + '\n')
	# fo.close()
	# global progress
	# progress.set('DONE! Output files written to\n' + getcwd())


# def func(args):
	# (p,q)=args
	# global x, y
	# x+=1
	# y+=10
	# print 'inside func: x = ', x
	# print 'inside fund: p = ', p, ', q = ', q
# 
# x=1
# y=2
# print 'before func: x = ', x, ', y = ', y
# func((12,14))
# print 'after func: x = ', x, ', y = ', y


# # ------------------------------------------------------------
# from __future__ import division
# import numpy as np
# import random

# N=1000; q=0.05;

# pv=np.empty(N)
# for i in range(N):
	# pv[i]=random.uniform(0,0.07)
# np.savetxt('pvals.txt',pv,fmt='%.8f',delimiter='\n')

# def fdr(P,q):
	# M=len(P)
	# I=np.argsort(P)
	# Ps=P[I]
	# Is=q*((1+np.arange(M,dtype='float'))/M)
	# H=np.zeros(M)
	# if np.any(Ps<=Is):
		# i_q=np.max(np.where(Ps<=Is)[0])
		# H[I[0:i_q+1]]=1
	# return H

# H = fdr(pv,0.05)
# np.savetxt('H.txt',H,fmt='%d',delimiter='\n')
# # ------------------------------------------------------------


# from Tkinter import *
# import tkMessageBox

# root = Tk()
# label = Label(root, text='heading')
# words = StringVar()
# words.set('My convention for regular Python programs taking all their input from the keyboard, and producing output displayed on a web page.\n\nThese programs can be run like other Python programs, directly from an operating system folder or from inside Idle.\n\nThey are not a final product, but are a way of breaking the development process into steps.')
# entry = Entry(root, textvariable=words)
# # scb = Scrollbar(root, orient='vertical')
# submit = Button(root, text='submit', command = lambda: tkMessageBox.showerror('ERROR',words.get()))

# label.grid(row=0,column=0)
# entry.grid(row=1,column=0)
# submit.grid(row=2,column=0)
# root.grid()

# root.mainloop()


# # ------------------------------------------------------------
# from __future__ import division
# from scipy import stats
# import numpy as np
# import sys

# out = np.loadtxt('dataset_practiceFinal.txt')
# Practice, Final = out[:,0], out[:,1]

# FinalC = Final - np.mean(Final)
# PracticeC = Practice - np.mean(Practice)

# def regress(x,y,order):
	# if (len(x)!=len(y)):
		# print "x and y are of unequal length! Quitting..."; sys.exit()
	# nobs=len(x); ncoef=order+1;
	# P=np.polyfit(x,y,order,full=True)
	# print "Coefficients ", P[0]
	# SSE = P[1][0]
	# SST = sum((y - np.mean(y))**2)
	# Rsq = (SST - SSE)/SST
	# adjRsq = 1-(1-Rsq)*((nobs-1)/(nobs-ncoef))
	# print 'Rsq = %g, adjRsq = %g\n\n' % (Rsq, adjRsq)
	# print "SSR = %g" % (SST - SSE)
	# X=np.column_stack((x,np.ones(len(x))))
	# yhat = np.sum(P[0]*X,1)
	# SSR = sum((yhat-np.mean(y))**2)
	# print "SSR calc = %g" % SSR


# print "---- Linear ----"
# regress(PracticeC,FinalC,1)

# # print "---- Quadratic ----"
# # regress(PracticeC,FinalC,2)

# # print "---- Cubic ----"
# # regress(PracticeC,FinalC,3)
# # ------------------------------------------------------------


# gradient, intercept, r_value, p_value, std_err = stats.linregress(PracticeC,FinalC)
# print "Gradient and intercept", gradient, intercept
# print "R-squared", r_value**2
# print "p-value", p_value
# R2 = r_value**2
# nobs = len(PracticeC)
# ncoef = 2
# adjR2 = 1-(1-R2)*((nobs-1)/(nobs-ncoef)) 
# print "adjusted R-squared", adjR2
# print "\n\n"


# import csv
# file=open('dataset_practiceFinal.csv')
# freader = csv.reader(file, delimiter='\t')
# for row in freader:
	# print row


# # ------------------------------------------------------------
# import xlrd
# import numpy as np

# # function to read the excel spreadsheet
# def xlsread(filename,nA,nB):
	# wb=xlrd.open_workbook(filename)
	# sh=wb.sheet_by_index(0)
	# # check if the number of columns are correct
	# if sh.ncols != 1+nA+nB:
		# sys.exit('incorrect number of columns in spreadsheet...')
	# # first column contains protein names
	# P=sh.col_values(0,1,)
	
	# # A = np.asmatrix(np.empty((len(P),nA)))
	# # for j in range(1,1+nA):
		# # A[:,j-1] = np.asmatrix(sh.col_values(j,1)).T
	# # B = np.asmatrix(np.empty((len(P),nB)))
	# # for j in range(1+nA,1+nA+nB):
		# # B[:,j-nA-1] = np.asmatrix(sh.col_values(j,1)).T

	# A=[]
	# for j in range(1,1+nA):
		# A.append(sh.col_values(j,1,))
	# B=[]
	# for j in range(1+nA,1+nA+nB):
		# B.append(sh.col_values(j,1,))
	# return P,np.column_stack(A),np.column_stack(B)

# P,A,B = xlsread('Cooper_147_vs_689.xls',3,3)
# fo = open('output.txt','w')
# fo.write('len(P) = %d, A.shape = (%d,%d), B.shape = (%d,%d)' % (len(P),A.shape[0],A.shape[1],B.shape[0],B.shape[1]))
# fo.close()
# print 'Done..\n'
# # ------------------------------------------------------------


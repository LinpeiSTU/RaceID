# coptright @linpei
# Email: linpei@shanghaitech.edu.cn
# data resource1 from:
#Single-cell messenger RNA sequencing reveals rare intestinal cell types, Dominic Gru, Anna Lyubimova, Lennart Kester, Kay Wiebrands, Onur Basak, Nobuo Sasaki, Hans Clevers & Alexander van Oudenaarden
#

# data resource2 from:
#Single-cell RNA-Seq profiling of human preimplantation embryos and embryonic stem cells,Liying Yan, Mingyu Yang, Hongshan Guo, Lu Yang, Jun Wu, Rong Li, Ping Liu, Ying Lian, Xiaoying Zheng, Jie Yan, Jin Huang, Ming Li, Xinglong Wu, Lu Wen, Kaiqin Lao, Ruiqiang Li, Jie Qiao & Fuchou Tang
#


import xlrd

import numpy as np
from numpy.random import random_sample

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import manifold, datasets,metrics,decomposition

from math import sqrt, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from time import time

import time
import hashlib
import scipy
from sklearn.datasets.samples_generator import make_blobs

import threading



def excel2matrix(path):#读excel数据转为矩阵函数
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows-1, ncols-1))
    for x in range(1,ncols):
        cols = table.col_values(x)
        cols=cols[1:]
        cols1 = np.matrix(cols)  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x-1] = cols1 # 把数据进行存储
    return datamatrix, nrows-1,ncols-1


datamatrix,nrows,ncols=excel2matrix('transcript_counts_intestine.xls')
#datamatrix,nrows,ncols=excel2matrix('test.xls')



def filterdata(datamatrix, ncols, nrows ,mintotal, minexpr, minnumber, maxexpr, normlog):
	#mintotal=3000, minexpr=5, minnumber=1, maxexpr=500, downsample=FALSE, dsn=1, rseed=17000) standardGeneric("filterdata")
	if mintotal<=0:
		print("mintotal has to be a positive number")
		exit(0)
	if minexpr<0:
		print("minexpr has to be a non-negative number")
		exit(0)
	if round(minnumber) != minnumber or minnumber < 0:
		print("minnumber has to be a non-negative integer number")
		exit(0)
	if type(normlog)!=bool :
		print("normlog has to be a bool value")
		exit(0)		

	col2delete=[]
	for i in range(ncols):
		if sum(datamatrix[:,i])<mintotal:
			col2delete.append(i)
	datamatrix=np.delete(datamatrix,[col2delete],axis=1)


	#normalization
	median=np.median(datamatrix.sum(axis=0))

# this is the method used in the raceID (original program), but the new program use log.
# log help to keep more character of the datas 
#	for i in range(datamatrix.shape[1]):
#		datamatrix[:,i]=datamatrix[:,i]/sum(datamatrix[:,i])*median
	
	if (normlog==True):
		np.log(datamatrix)

		row2delete=[]
		for i in range(nrows):
			if (datamatrix[i,:] >=np.log(minexpr)).sum()<  minnumber:
				row2delete.append(i)
			if max(datamatrix[i,:])>np.log( maxexpr):
				row2delete.append(i)
		datamatrix=np.delete(datamatrix,row2delete,axis=0)

	else:
		for i in range(datamatrix.shape[1]):
			datamatrix[:,i]=datamatrix[:,i]/sum(datamatrix[:,i])*median

		row2delete=[]

		for i in range(nrows):
			if (datamatrix[i,:] >=minexpr).sum()<minnumber:
				row2delete.append(i)
			if max(datamatrix[i,:])>maxexpr:
				row2delete.append(i)
		datamatrix=np.delete(datamatrix,row2delete,axis=0)

	
	return datamatrix




datamatrix=filterdata(datamatrix,ncols ,nrows,3000,5,1,500,False)



#  change  np.matrix  to dataframe , which is needed in the following function below
#  if for the data , colmun is the cell-id  & row is the RNA ===> form==True   ===> transform
#   C1 C2 C3 C4 C4
# R1
# R2
# R3
# R4
# R5
#  if for the data , colmun is the RNA  &  row id the cell-id ===> form==False    ===> no transform
#   R1 R2 R3 R4 R4
# C1
# C2
# C3
# C4
# C5
def transform2dataframe(datamatrix, form):
	if type(form)!=bool:
		print("form has to be in boolean type")
		exit(0)

	if form==True:
		datamatrix = pd.DataFrame(datamatrix.T, columns=None)
	else:
		datamatrix = pd.DataFrame(datamatrix, columns=None)

	return datamatrix

datamatrix=transform2dataframe(datamatrix, True)




#data form should be in dataframe
# to find the best k in the kmeans, many methods are supported
# in this program, " GAP STATISTIC " is used to estime the best K
# also will plot the (k, gap) plot to users
def gap_statistic(data, nrefs=5, maxClusters=20):

#nrefs: number of sample reference datasets to create
    gaps=[]

    for i in range(1,maxClusters+1):

    	#record the distance W_0 by random data
    	#Gap_statistic use the Monte-corle method the estimate the outcome
        refDisps=[]        
        for j in range(nrefs):

        	#to generate the random dataset with the same shape as the data of input
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(i)
            km.fit(randomReference)

            # to get the distance W
            refDisp = km.inertia_
            refDisps.append(refDisp)

        # to get the distance W_1 of the origin data
        km = KMeans(i)
        km.fit(data)
        origDisp = km.inertia_

        # GAP(k)=E(log(W_0))-log(W_1)
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps.append(gap)
    plt.plot(range(1,maxClusters+1), gaps, linewidth=3)
    plt.scatter(gaps.index(max(gaps))+1, max(gaps), s=250, c='r')
    plt.grid(True)
    plt.title('Gap Value')
    plt.xticks(np.arange(2, 21, 1))
    plt.xlabel('Number of Cluster')
    plt.ylabel('Gap Value')
    plt.show()
    
    return gaps.index(max(gaps))+1

#gap_statistic(datamatrix)




#USE silhouette to evaluate the performance of kmeans,
# silhouette method has already supported by a function in sklearn.metrics.silouette()
#  in put the K and data , will out put the performance  of K
def silhouette(max_K,data):
	s=[]
	# can be stored in the np array form
	for k in range(2,max_K+1):
	    kmeanModel = KMeans(n_clusters = k, random_state = 0).fit(data)
	    labels = kmeanModel.fit_predict(data)
	    s.append(metrics.silhouette_score(data, labels, metric = 'euclidean'))

	plt.plot(range(2,max_K+1),s , linewidth=3)
	plt.grid(True)
	plt.title('Silhoutte Score')
	plt.xticks(np.arange(2, max_K+1, 1))
	plt.xlabel('Number of Cluster')
	plt.ylabel('Silhoutte value')
	plt.show()
#	plt.plot(range(2,max_k+1),s, 'bx-')
#	plt.show()
	return s




#print(silhouette(20,datamatrix))








#############################################
#       
# DIMENSION REDUCTION
#
############################################


#tSNE
#data should be in the form of np.array
def dim_reduce_tsne(data,dimension,perplexity):
	X=data.values
	tsne = manifold.TSNE(n_components=dimension, init='random', perplexity=perplexity)
	#tsne = manifold.TSNE(n_components=n_components, init='random',random_state=0, perplexity=perplexity)
	X= tsne.fit_transform(X)
	return X

X=dim_reduce_tsne(datamatrix,2,29.2)


# PCA
##data should be in the form of np.array
def dim_reduce_pca(data,dimension):
	Y=data.values
	pca=decomposition.PCA(n_components=dimension, copy=True, whiten=False)
	Y=pca.fit_transform(Y)
	return Y
#X=dim_reduce_pca(datamatrix,2)

#plot for kmeans with dimension 2 and the data should be in 2 dimension, 
#data should be out put from dim_reduce_tsne() or  dim_reduce_pca()
def plot_kmeans_di2(data,max_k):
# plot series of figures for k=2,3,4,5,6,7,8,9
	if data.shape[1]!=2:
		print("this function is only allowed for the 2-dim data, please use dim_reduce_tsne() or dim_reduce_pca()")
		exit(0)

	for i in range(2,max_k+1):
		plt.subplot(3,3,i-1)
		y_pred = KMeans(n_clusters=i, random_state=0).fit_predict(data)
		plt.scatter(data[:, 0], data[:, 1], c=y_pred)
		plt.title("Figure for K=%d " %i)
	plt.show()
	return 1

plot_kmeans_di2(X,10)








######################################################
#this part is designed for optimization and accelerating
#
# to accelerate the calculate , First I use THREADING , which is multi-thread inner package for the python programming
# choose this function seriously, in different environment, the time to finish can be varified
#
#
#######################################################

#######################################################################################
# this is for accelerating the silhouette score
class SilhoutteThread (threading.Thread):
    def __init__(self, threadID, name,data,k,array):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data=data
        self.k=k
        self.array=array
    def run(self):
        sil_score=single_silhouette(self.data, self.k)
        self.array.append((self.k,sil_score))

def single_silhouette(data, k):
    kmeanModel = KMeans(n_clusters = k, random_state = 0).fit(data)
    labels = kmeanModel.fit_predict(data)
    s=metrics.silhouette_score(data, labels, metric = 'euclidean')
    
    return s

#######################
#this is the main function and only call this function can satisfy the need,`
#by default, the number of threads are 10, change it as you like
# return the silhouette score and also plot the plot
silhouette_score=[]
def threadForSilhouette(data):
    global silhouette_score
    silhouette=[]
    thread1 = SilhoutteThread(1, "Thread-1",data,2,silhouette_score)
    thread2 = SilhoutteThread(2, "Thread-2",data,3,silhouette_score)
    thread3 = SilhoutteThread(3, "Thread-3",data,4,silhouette_score)
    thread4 = SilhoutteThread(4, "Thread-4",data,5,silhouette_score)
    thread5 = SilhoutteThread(5, "Thread-5",data,6,silhouette_score)
    thread6 = SilhoutteThread(6, "Thread-6",data,7,silhouette_score)
    thread7 = SilhoutteThread(7, "Thread-7",data,8,silhouette_score)
    thread8 = SilhoutteThread(8, "Thread-8",data,9,silhouette_score)
    thread9 = SilhoutteThread(9, "Thread-9",data,10,silhouette_score)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    thread9.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()
    thread9.join()

    silhouette_score.sort(key=lambda x:x[0])

    x=[i[0] for i in silhouette_score ]
    y=[i[1] for i in silhouette_score]

    plt.plot(x,y, linewidth=3)
    plt.grid(True)
    plt.title('Silhoutte Score')

    plt.xticks(np.arange(2, 11, 1))
    plt.xlabel('Number of Cluster')
    plt.ylabel('Silhoutte value')
    plt.show()
    return silhouette_score

#silhouette_score =threadForSilhouette(datamatrix)






##################################################################################################
# this is for accelerating the silhouette score
class GapThread (threading.Thread):
    def __init__(self, threadID, name,data,k,array):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data=data
        self.k=k
        self.array=array
    def run(self):
        gap=single_gap_statistic(self.data, self.k)
        self.array.append([self.k,gap])




#######################
#this is the main function and only call this function can satisfy the need,
#by default, the number of threads are 15, change it as you like
# return the silhouette score and also plot the plot
def single_gap_statistic(data, k,nrefs=5):
#nrefs: number of sample reference datasets to create
        #record the distance W_0 by random data
        #Gap_statistic use the Monte-corle method the estimate the outcome
        refDisps=[]        
        for j in range(nrefs):

            #to generate the random dataset with the same shape as the data of input
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(k)
            km.fit(randomReference)

            # to get the distance W
            refDisp = km.inertia_
            refDisps.append(refDisp)

        # to get the distance W_1 of the origin data
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_

        # GAP(k)=E(log(W_0))-log(W_1)
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        return gap


#######################
#this is the main function and only call this function can satisfy the need,
#by default, the number of threads are 15, change it as you like
# return the silhouette score and also plot the plot
gap_score=[]
def threadForGap(data):
    global gap_score
    gap_score=[]
    thread1 = GapThread(1, "Thread-1",data,2,gap_score)
    thread2 = GapThread(2, "Thread-2",data,3,gap_score)
    thread3 = GapThread(3, "Thread-3",data,4,gap_score)
    thread4 = GapThread(4, "Thread-4",data,5,gap_score)
    thread5 = GapThread(5, "Thread-5",data,6,gap_score)
    thread6 = GapThread(6, "Thread-6",data,7,gap_score)
    thread7 = GapThread(7, "Thread-7",data,8,gap_score)
    thread8 = GapThread(8, "Thread-8",data,9,gap_score)
    thread9 = GapThread(9, "Thread-9",data,10,gap_score)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    thread9.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()
    thread9.join()

    gap_score.sort(key=lambda x:x[0])

    x=[i[0] for i in gap_score ]
    y=[i[1] for i in gap_score]

    plt.plot(x,y , linewidth=3)
    plt.grid(True)
    plt.title('Gap Score')
    plt.xticks(np.arange(2, 11, 1))
    plt.xlabel('Number of Cluster')
    plt.ylabel('gap value')
    plt.show()
    

    return gap_score











# to accelerate the calculate , Second I use pyspark , which is multi-process construction for parrelization
# choose this function seriously, in different environment, the time to finish can be varied,
# but in most condition, it works well.




































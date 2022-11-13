#Laszlo Frazer 2022
#Clustering tool
#
#Reads in time series and edivisive change point calculation
#Time series is subdivided by change points
#Computes separation metric for each pair of subdivisions
#Computes mean of means for each pair of subdivisions
#Computes product of subdivision lengths
#Goal is to create a graphic that visualizes clustering

#Computes maximum number of clusters with DBSCAN

import csv
import sys
import math
import itertools
import numpy
from itertools import tee, islice, chain, combinations
from statistics import mean, stdev
from sklearn.cluster import DBSCAN

#if you need to access current element and next element of list
#probably unnecessary complication
def getpairs(some_iterable):
    items, nexts = tee(some_iterable, 2) #selects pairs
    nexts = chain(islice(nexts, 1, None), [None]) #makes nexts only second item in pair
    #None indicates invalid data
    return zip(items, nexts)

filenumber=sys.argv[1]
timeseriesf=open(filenumber+".brightness.q")
edivf=open("changepoints/"+filenumber+".ediv")
timeseriescsv=csv.reader(timeseriesf)
edivcsv=csv.reader(edivf)

timeseries=[]#the data
ediv=[]#edivisive changepoint partititions
metrics=[]#sum of pairwise differences of two partitions
means=[]#mean of the means of the two partitions
variance=[] #geometric mean of the standard deviations of the two series
lengths=[]#product of the lengths of the two partitions
ranges=[]#re-output ediv in a convenient format

for row in timeseriescsv:
    timeseries.append(float(row[0]))

for row in edivcsv:
    ediv.append(int(row[0]))

print("The number of partition pairs is: "+str(math.comb(len(ediv)-3,2)))  
#Tells us how big the calculation will be
#Do not include the two partitions at the end and one at the beginning
#first partition does not have a valid start point
#last ediv partition does not have any data at all
#second-to-last ediv partition is the start of a partition with an invalid endpoint

for i in range(len(ediv)-1):
    if (3>(ediv[i+1]-ediv[i])):
        print("Warning: Not enough data to compute standard deviation in edivisive subdivision ",ediv[i],", ",ediv[i+1],".",sep="")

#produce ordered pairs of subdivisions
#where each ordered pair includes the start point and the end point.
subdivcombs=combinations(getpairs(ediv[1:-1]),2)
for i in subdivcombs: #i is an ordered pair of ordered pairs
    if (None!=i[1][1]): #skip situations where endpoints are beyond the range of the data
        metr=0#reset

        #loop over both partitions 
        for k in timeseries[i[0][0]-1:i[0][1]-2]:#first partition 
            #python is zero indexed, edivisive is not, so start needs to have 1 subtracted
            #end is one before the next partition, so end needs 2 subtracted
            #partitions have at least one element
            for l in timeseries[i[1][0]-1:i[1][1]-2]:#second partition
                metr=metr+abs(k-l) #compute the sum of absolute differences of all pairs of elements
        metrics.append(metr/((i[0][1]-i[0][0]-1)*(i[1][1]-i[1][0]-1)))  #average those differences by dividing by the number of pairs 
        means.append(mean([mean(timeseries[i[0][0]-1:i[0][1]-2]),mean(timeseries[i[1][0]-1:i[1][1]-2])])) #mean of mean of the two partition
        ranges.append([i[0][0],i[0][1]-1,i[1][0],i[1][1]-1]) #re-output ediv in a convenient format

        #variance
        if((3<(i[0][1]-i[0][0])) and (3<(i[1][1]-i[1][0]))):
            variance.append(mean([stdev(timeseries[i[0][0]-1:i[0][1]-2]),stdev(timeseries[i[1][0]-1:i[1][1]-2])])) #mean of the standard deviation of the two partitions
        else:
            variance.append("NaN")

        lengths.append((i[0][1]-i[0][0])*(i[1][1]-i[1][0]))  #product of lengths of two partitions.  length is start of next minus start of current 
        #length is also the number of pairs of time point elements

with open("cluster/"+filenumber+".cluster",'w') as f:
    csv.writer(f,delimiter='\t').writerows(itertools.zip_longest(metrics,means,variance,lengths,[i[0] for i in ranges],[i[1] for i in ranges],[i[2] for i in ranges],[i[3] for i in ranges]))


#setup for DBSCAN clustering algorithm
#generate matrix filled with zeros
#omit first and last due to invalid data
#omit second to last because it is a partition endpoint
matrmetric=numpy.zeros((len(ediv)-3,len(ediv)-3))
segmeans=[]

#rather than reuse the previously computed metric, I decided it was easier just to recompute it
for i in range(len(ediv)-3):
    segmeans.append(mean(timeseries[(ediv[i+1]-1):(ediv[i+2]-2)]))#mean of a partition
    for j in range(len(ediv)-3):
            if(i>j):#leave diagonal zero
                metr=0

                #for every pair of elements in the two partitions
                #compute the absolute value of their difference
                #python is zero indexed, edivisive is not, so start needs to have 1 subtracted
                #end is one before the next partition, so end needs 2 subtracted
                #partitions have at least one element
                for k in timeseries[(ediv[i+1]-1):(ediv[i+2]-2)]:
                    for l in timeseries[(ediv[j+1]-1):(ediv[j+2]-2)]:
                        metr=metr+abs(k-l) #add difference of elements k and l

                #weight the average, divide by number of pairs 
                metr=metr/((ediv[i+2]-ediv[i+1]-1)*(ediv[j+2]-ediv[j+1]-1))
                #make the matrix symmetric
                matrmetric[i,j]=metr
                matrmetric[j,i]=metr
        

#loop over maximum separation; cluster for each option
#crudely look for smallest maximum separation that gives the maximum number of clusters
#we are testing systematically instead of optimizing because we want to see the 
#relationship between separation limit and number of clusters
#this will exclude as much noise as possible from the clusters
start=min(metrics)
end=max(metrics)
separationl=[]
maxclusterl=[]
uncategorizedl=[]
granularity=1000#granularity of systematic testing
cutoff=.3 #usually no point in calculating DBSCAN epsilon values more than 
#.3 of the range of metrics

assert(sorted(uncategorizedl)==uncategorizedl[::-1]) #ensure that as epsilon increases, uncategorized partitions decreases

for i in range(1,granularity):
    separation=cutoff*(end-start)*i/granularity
    #compute clusters from precomputed metric matrix
    cluster=DBSCAN(eps=separation,min_samples=2,metric='precomputed').fit_predict(matrmetric)
    #count number of partitions in each cluster
    unique, counts = numpy.unique(cluster, return_counts=True)

    #determine how many partitions are not clustered
    if(-1==unique[0]):
        uncategorized=counts[0]
    else: 
        assert(-1<min(unique))
        uncategorized=0
    separationl.append(separation)
    maxclusterl.append(max(cluster)) #if no clusters, max is -1
    uncategorizedl.append(uncategorized)

with open("cluster/"+filenumber+".sep",'w') as f:
    csv.writer(f,delimiter='\t').writerows(itertools.zip_longest(separationl,maxclusterl,uncategorizedl))
    #when plotting, add one to maximum number of clusters because -1 indicates all segments are unclustered


#verify we can compute the location of the eps parameter of DBSCAN where the number of clusters reaches it's maximum
assert(max(maxclusterl)==maxclusterl[maxclusterl.index(max(maxclusterl))])
#I am choosing that as the best eps.
#add 1 because python is zero indexed but the separation loop above starts with 1
separation=cutoff*(end-start)*(maxclusterl.index(max(maxclusterl))+1)/granularity
#this is the optimal clustering, in my opinion
cluster=DBSCAN(eps=separation,min_samples=2,metric='precomputed').fit_predict(matrmetric)
print(separation,max(cluster))

#save clusters
with open("cluster/"+filenumber+".dbscan",'w') as f:
    csv.writer(f,delimiter='\t').writerows(itertools.zip_longest(cluster,ediv[1:-2],[x-1 for x in ediv[2:-1]],segmeans))
#columns: cluster, segment start location, segment end location


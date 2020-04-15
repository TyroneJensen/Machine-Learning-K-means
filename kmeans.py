#This program uses a K-Means clustering implementation to visualise possible relationships and categorise into clusters the unlabelled input data 

#import modules
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import random


#function that computes the distance between two data points
def dist (x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)
   
    
# function that reads data in from the csv files  
data= []

def input_reader ():
    file = input ('Enter data file (include file extension):')
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            data.append(row)
            
input_reader()        


#create a panda data frame
df = pd.DataFrame(data, columns=["Country", "x", "y"])
#cast variables to type float
df["x"], df["y"] = df["x"].astype(float), df["y"].astype(float)  


#initialising kmeans algorithm
#==============================

#Determining the number of clusters and iterations
clusters = int(input("Number of clusters? "))
iters = int(input("Number of iterations? "))


#Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
centroids = []
#create random array
index = random.sample(range(196), clusters)
for i in range(clusters):   
    #get x,y co-ordinates for each centroid 
    centroids.append(df["x"][index[i]])
    centroids.append(df["y"][index[i]])
#reshape centroid array data to have 2 columns
centroids = np.reshape(centroids, (clusters, 2))


#a function to determine distance and closeness between data points and centroids
def dist_and_closest(df, centroids):
    
    distances = []
    #determines distance between data point and centroid
    for i in range(len(centroids)): 
        for x in range(len(df)):    
            distances.append(dist(df["x"][x], df["y"][x], centroids[i][0], centroids[i][1]))
            df[i] = pd.Series(distances)
        distances = []
        
    #Assign each data point to the closest cluster (centroid).
    closest = []
    for i in range(len(df)):
        min_col = df[0][i]
        for x in range(1, len(centroids)):
            min_col = np.min((min_col, df[x][i]))
        closest.append(df.iloc[i].eq(min_col).idxmax())
    df["closest"] = pd.Series(closest)


#Implement kmeans algorithm and plot points
#==========================================
    
total_dist = 0

for i in range(iters):
    
    plt.ylabel("Life Expectancy in Years")
    plt.xlabel("Births per 1000 People Each Year")
    
    #first iteration
    if(i == 0):
        
        #get starting centroids, distances and closest
        dist_and_closest(df, centroids)
        
        #plot centroids
        for x in range(len(centroids)):
            plt.scatter(centroids[x][0], centroids[x][1], marker='+', s=200, c='r')
        #plot data
        for x in range(len(df)):
            plt.scatter(df["x"][x], df["y"][x], marker='.', c='b')
        #visualize figure
        plt.show()
        
        #calculates sum of all data point distances between centroid
        for y in range(len(centroids)):
            for z in range(len(df)):
                if(df["closest"][z] == y):
                    total_dist = total_dist + df[y][z]
        
        print("Sum of all distances from each data point to its cluster mean: ", total_dist)
        
        total_dist = 0
        
    #middle iterations
    elif(i > 0) and (i < iters-1):
        
        meanx, meany, count = 0, 0, 0
        
        #Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster
        for x in range(len(centroids)):
            for y in range(len(df)):
                if(df["closest"][y] == x):
                    count = count + 1
                    meanx = meanx + df["x"][y]
                    meany = meany + df["y"][y]
            centroids[x][0] = meanx/count
            centroids[x][1] = meany/count
            count, meanx, meany = 0, 0, 0
        
        dist_and_closest(df, centroids)
        
        #plot centroids
        for x in range(len(centroids)):
            plt.scatter(centroids[x][0], centroids[x][1], marker='+', s=150, c='r')
        #plot data
        for x in range(len(df)):
            plt.scatter(df["x"][x], df["y"][x], marker='.',c='b') 
        #visualize figure
        plt.show()
        
        #calculates sum of all data point distances between centroid
        for y in range(len(centroids)):
            for z in range(len(df)):
                if(df["closest"][z] == y):
                    total_dist = total_dist + df[y][z]
        
        print("Sum of all distances from each data point to its cluster mean: ", total_dist)
        
        total_dist = 0
        
    #last iteration  
    else:
         
        count_list = []
        
        meanx, meany, count = 0, 0, 0
        
        #Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster
        for x in range(len(centroids)):
            for y in range(len(df)):
                if(df["closest"][y] == x):
                    count = count + 1
                    meanx = meanx + df["x"][y]
                    meany = meany + df["y"][y]
            centroids[x][0] = meanx/count
            centroids[x][1] = meany/count
            
            count, meanx, meany = 0, 0, 0
            
            dist_and_closest(df, centroids)
         
        #plot centroids    
        for x in range(len(centroids)):
            plt.scatter(centroids[x][0], centroids[x][1], marker='+', s=150, c='r')
        #plot data
        for x in range(len(df)):
            plt.scatter(df["x"][x], df["y"][x], marker='.', c='b')
       
        
        for x in range(len(centroids)):
            for y in range(len(df)):
                if(df["closest"][y] == x):
                    count = count + 1
                    meanx = meanx + df["x"][y]
                    meany = meany + df["y"][y]
            centroids[x][0] = meanx/count
            centroids[x][1] = meany/count
            
            count_list.append(count)
            
            count, meanx, meany = 0, 0, 0
            

            
        # Print out the results
        
            #print number of countries in each cluster 
            print("The number of countries in cluster {} are:".format(x+1))
            print(count_list[x])
            print()
            #print births per 1000 for cluster
            print("The mean number of births per 1000 people each year in cluster {} is:".format(x+1))
            print(centroids[x][0])
            print()
            #print life expec. for cluster 
            print("The mean life expectancy in years in cluster {} is:".format(x+1))
            print(centroids[x][1])
            print()
            #print list of countries in cluster
            print("All the countries in cluster {}:".format(x+1))
            for y in range(len(df)):
                if(df["closest"][y] == x):
                    print(df["Country"][y])
            print()
                    
    
        #visualise figure    
        plt.show() 
        #calculates sum of all data point distances between centroid
        for y in range(len(centroids)):
            for z in range(len(df)):
                if(df["closest"][z] == y):
                    total_dist = total_dist + df[y][z]
        print("Sum of all distances from each point to its cluster mean: ", total_dist)
        total_dist = 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
class NBA:
    def __init__(self):
        self.df = pd.read_csv("datasets_1358_30676_player_data.csv")
        self.df.dropna(inplace = True)
        self.x   = self.df['height'].apply(self.parseColumn)
        self.X = np.array(list(zip(self.x, self.df["weight"])))
        self.numCent = 2

    def parseColumn(self,x):
        ft,sep,inch = x.partition("-")
        convFactor = .0254
        inch = int(ft)*12+int(inch)
        return round(inch*convFactor,3)
    def scatter_data(self,x,y):
        C = self.create_cent()
        Cx,Cy = zip(*C)
        plt.scatter(self.df[x],self.df[y], c ='b', s = 18)
        plt.scatter(Cx,Cy ,marker ="+", c ='r', s =160)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
    def create_cent(self):
        Cx = np.random.randint(np.min(self.X[:,0]),np.max(self.X[:,0]), size = self.numCent)
        Cy = np.random.randint(np.min(self.X[:,1]),np.max(self.X[:,1]), size = self.numCent)
        C = np.array(list(zip(Cx,Cy)), dtype =np.float64)
        return C

    def do_Scki(self,x,y):
        kMeans = KMeans(n_clusters = self.numCent)
        kMeans = kMeans.fit(self.X)
        cent = kMeans.cluster_centers_
        c = ['b','y','r','g','c','m']
        labels = kMeans.predict(self.X)
        color = [c[i] for i in labels]
        fig, ax = plt.subplots()
        ax.set_xlabel("Height in meters")
        ax.set_ylabel("Weight")
        plt.scatter(self.X[:,0],self.X[:,1] ,c = color, s= 18)
        plt.scatter(cent[:,0], cent[:,1], marker = "+", s =100, c = 'black')
        plt.show()
        print(cent)
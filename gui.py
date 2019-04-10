from tkinter import *
import tkinter
import pygame
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

my_cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])

my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","purple","green"])

pygame.init()
WIDTH=800
WIDTHGAME=400
HEIGHT=500
WHITE=(255,255,255)
BLUE=(0,0,255)
RED=(255,0,0)
GREEN=(0,255,0)
BOUN=(165,42,42)

window=tkinter.Tk()
window.title("Work with Neural Net")
window.geometry('500x300')
#label=tk.Label(window,text="Welcome to the module, Please enter details").pack()
l1=Label(window,text="Welcome to the module, Please enter details",font=("Arial Bolad",12))
l1.grid(column=0,row=0)
l2=Label(window,text="Number of data points",font=("Arial Bolad",10))
l2.grid(column=0,row=4)
txt1=Entry(window,width=10)
txt1.grid(column=0,row=5)

l3=Label(window,text="Number of epochs",font=("Arial Bolad",10))
l3.grid(column=0,row=6)
txt2=Entry(window,width=10)
txt2.grid(column=0,row=7)

l3=Label(window,text="Number of hidden layers with space  eg  2 3 ",font=("Arial Bolad",8))
l3.grid(column=0,row=8)
txt3=Entry(window,width=10)
txt3.grid(column=0,row=9)

l7=Label(window,text="Learning Rate",font=("Arial Bolad",8))
l7.grid(column=0,row=10)
txt4=Entry(window,width=10)
txt4.grid(column=0,row=11)

class FFSN_MultiClass:
    def __init__(self, n_inputs, n_outputs, hidden_sizes=[3]):
        print("INITIALISING")
        self.nx = n_inputs
        self.ny = n_outputs
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny] 

        self.W = {}
        self.B = {}
        for i in range(self.nh+1):
            self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
            self.B[i+1] = np.zeros((1, self.sizes[i+1]))
        print("INITIALIZED")
          
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
  
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]
            self.H[i+1] = self.sigmoid(self.A[i+1])
        self.A[self.nh+1] = np.matmul(self.H[self.nh], self.W[self.nh+1]) + self.B[self.nh+1]
        self.H[self.nh+1] = self.softmax(self.A[self.nh+1])
        return self.H[self.nh+1]
  
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()
  
    def predict2(self,x1,x2):
        xzz=np.array([[x1,x2]])
        y_pred=self.forward_pass(xzz)
        return np.array(y_pred).squeeze()
 
    def grad_sigmoid(self, x):
        return x*(1-x)
    
    def cross_entropy(self,label,pred):
        yl=np.multiply(pred,label)
        yl=yl[yl!=0]
        yl=-np.log(yl)
        yl=np.mean(yl)
        return yl
 
    def grad(self, x, y):
        self.forward_pass(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.nh + 1
        self.dA[L] = (self.H[L] - y)
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)
            self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1])) 
    
    def fit(self, X, Y, epochs, initialize='True', learning_rate=0.01, display_loss=False):
        if display_loss:
            loss = {}
        if initialize:
            for i in range(self.nh+1):
                self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
                self.B[i+1] = np.zeros((1, self.sizes[i+1]))
              
        for epoch in range(epochs):
            print(epoch)
            dW = {}
            dB = {}
            for i in range(self.nh+1):
                dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))
                dB[i+1] = np.zeros((1, self.sizes[i+1]))
            for x, y in zip(X, Y):
                self.grad(x, y)
                for i in range(self.nh+1):
                    dW[i+1] += self.dW[i+1]
                    dB[i+1] += self.dB[i+1]

            m = X.shape[0]
            for i in range(self.nh+1):
                self.W[i+1] -= learning_rate * (dW[i+1]/m)
                self.B[i+1] -= learning_rate * (dB[i+1]/m)

            if display_loss:
                Y_pred = self.predict(X)
                loss[epoch] = self.cross_entropy(Y, Y_pred)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('CE')
            plt.show()
      
    def showplot(self,X,layer,num):
        Y_pred=[]
        for x in X:
            self.forward_pass(x);
            Y_pred.append(self.H[layer].squeeze()[num])
        return np.asarray(Y_pred)
            
        



class Neuron:
    def __init__(self):
        self.x=(int)(WIDTHGAME/2)
        self.size=10
        self.layer=0
        self.num=0
        self.color=(0,0,0)
    def poss(self,x,y):
        self.x=x
        self.y=y
    def chc(self,n):
        if(n==0):
            self.color=GREEN
        elif(n==1):
            self.color=BLUE




def clicked():
    np.random.seed(0)
    f=txt3.get()
    ep=txt2.get()
    lr=float(txt4.get())
    
    ll=list((f.split()))
    ll=list(map(int,ll))
    h_layers=len(ll)
    origll=ll[:]
    ll=[2]+ll
    ll.append(2)
    print(ll)
    total_n=0
    for t in ll:
        total_n=total_n+t

    data, labels = make_blobs(n_samples=int(txt1.get()), centers=4, n_features=2, random_state=0)
    for i in range(labels.shape[0]):
        labels=labels%2
    
    l5=Label(window,text="Close the dialogue box to proceed",font=("Arial Bolad",10))
    l5.grid(column=0,row=13)   
    plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap2)
    plt.show()
    
    enc = OneHotEncoder()
    y_labels= enc.fit_transform(np.expand_dims(labels,1)).toarray()
    ffn=FFSN_MultiClass(2,2,origll)
    ffn.fit(data,y_labels,epochs=int(ep),initialize='True', learning_rate=lr, display_loss=True)
    
    game_display=pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("Your Network")
    clock=pygame.time.Clock()

    objs=[Neuron() for i in range(total_n)]

    last=0

    layn=0  
    for t in ll:
        for m in range(t):
            objs[m+last].layer=layn
            objs[m+last].num=m
            if(layn==0):
                objs[m+last].chc(0)
            elif(layn==(h_layers+1)):
                objs[m+last].chc(1)
            objs[m+last].poss((m+1)*(WIDTH-20)/(t+1)+10, (HEIGHT-objs[m+last].size)-(layn)*(HEIGHT-2*objs[m+last].size)/(h_layers+1))
        layn=layn+1
        last+=t

    def draw_enviroment(objs):
        game_display.fill(WHITE)
        for obj in objs:
            pygame.draw.circle(game_display,obj.color,[(int)(obj.x),(int)(obj.y)],obj.size)

        ttt=h_layers+1
        for i in range(total_n-1):
            for j in range(i+1,total_n):
                if(objs[j].layer==objs[i].layer+1):
                    pygame.draw.line(game_display,(100,100,100),(objs[i].x,objs[i].y),(objs[j].x,objs[j].y))
        pygame.display.update()

    def find(pos):
        for i in range(2,total_n):
            if((pos[0]-objs[i].x)**2+(pos[1]-objs[i].y)**2<objs[i].size*objs[i].size):
                TOWL=objs[i].layer
                TOWN=objs[i].num
                print(TOWL,TOWN)

                x_min, x_max = data[:,0].min() - 0.5, data[:,0].max() + 0.5
                y_min, y_max = data[:,1].min() - 0.5, data[:,1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),np.arange(y_min, y_max, 0.2))
                fig, ax = plt.subplots(figsize=(10,5))
                Z = np.array(ffn.showplot(np.c_[xx.ravel(), yy.ravel()],TOWL,TOWN))
                Z = Z.reshape(xx.shape)
                ax.scatter(data[:,0],data[:,1],c=labels,cmap=my_cmap2)
                ax.contourf(xx, yy, Z,cmap=my_cmap2, alpha=0.2)
                plt.show()


    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                quit()
            if event.type==pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    pos=pygame.mouse.get_pos()
                    find(pos)
        draw_enviroment(objs)
        clock.tick(60)
        


    
        
    
    

    














    
    
    
    
    
    











bt=Button(window,text="Submit",command=clicked)
bt.grid(column=0,row=12)
window.mainloop()

import torch,torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import copy 

#from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

import numpy as np
import pygame
import random
import math
import time
from model import Linear_QNet, QTrainer


#train data is n0 and label data is n1


#def create_lenet():
    #model = Linear_QNet(10000,400,10000)
   # return model

def create_lenet():
    model = nn.Sequential(
        

        #nn.Flatten(),
        nn.Linear(8,256),
        nn.ReLU(),
        nn.Linear(256,100),
        
        nn.ReLU(),
        nn.Linear(100,1)
    )
    return model




pygame.init()
temp_arr = []
WIDTH, HEIGHT = 800, 500#size of display(can be changed but remember to change count value)
size = 5
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
board = []
for i in range(0,int(WIDTH/size)):
    row = []
    for j in range(0,int(HEIGHT/size)):
        if i*j%6 == 0:
            #color = (208,0,0)
            row.append(1)
        else:
            #color = (3,176,26)
            row.append(0)
        #pygame.draw.rect(WIN, color, pygame.Rect(i*size, j*size, i+size, j+size))#draws the squres
    board.append(row)
prev_board = []
def next_gen():
    global board
    board = np.array(board)
    board = board.reshape(int(WIDTH/size),int(HEIGHT/size))
    temp_board = []
    for i in range(0,len(board)):
        temp_board_row = []
        for j in range(0,len(board[i])):
            neighbours = 0
            try:
                if board[i-1][j-1] == 1:
                    neighbours += 1
            except:
                pass
            try:
                if board[i][j-1] == 1:
                    neighbours += 1
            except:
                pass
            try:
                if board[i+1][j-1] == 1:
                    neighbours += 1
            except:
                pass
            try:
                if board[i-1][j] == 1:
                    neighbours += 1
            except:
                pass
            try:
                if board[i+1][j] == 1:
                    neighbours += 1#
            except:
                pass
            try:
                if board[i][j+1] == 1:
                    neighbours += 1
            except:
                pass
            try:
                if board[i+1][j+1] == 1:
                    neighbours += 1
            except:
                pass
            try:
                if board[i-1][j+1] == 1:
                    neighbours += 1
            except:
                pass

            if neighbours >= 3 and board[i][j] == 0:
                temp_board_row.append(1)
            if neighbours < 3 and board[i][j] == 0:
                temp_board_row.append(0)

            if neighbours < 2 and board[i][j] == 1:
                temp_board_row.append(0)

            if neighbours > 3 and board[i][j] == 1:
                temp_board_row.append(0)

            if neighbours == 2 and board[i][j] == 1:
                temp_board_row.append(1)
            if neighbours == 3 and board[i][j] == 1:
                temp_board_row.append(1)
            

        temp_board.append(temp_board_row)
    for i in range(0,len(temp_board)):
        
        for j in range(0,len(temp_board[i])):
            if temp_board[i][j] == 0:
                color = (3,0,26)
            else:
                color = (3,176,26)
            #pygame.draw.rect(WIN, color, pygame.Rect(i*size, j*size, i+size, j+size))#draws the squres
    global prev_board
    global temp_arr
    if temp_arr != []:
        prev_board = temp_arr
        
    else:
        prev_board = board
    for i in range(0,len(board)-1):
        for j in range(0,len(board[i])-1):

            board[i][j] = temp_board[i][j]
    board = np.array(board)
    board = board.reshape(int((WIDTH/size)*(HEIGHT/size)))
    prev_board = np.array(prev_board)
    prev_board = prev_board.reshape(int((WIDTH/size)*(HEIGHT/size)))
out_arr = []
out_pred = []
max_accuracy = 0
best_model = ''
tensorboard = []
tensorprev_board = []
def validate(model, data):
    global prev_board
    global board
    global tensorprev_board
    global tensorboard
    total = 0
    correct = 0
    #for i, (images, labels) in enumerate(data):
        #images = images.cuda()
    x = model(data)
    value, pred = torch.max(x,0)
    
    pred = pred.data.cpu()
    
    total += x.size(0)
    correct += torch.sum(data == tensorboard)
    return correct*100./total

def train(lr=1e-3, device="cpu",epoch=0):
    global prev_board
    global board
    global tensorprev_board
    global tensorboard
    global out_pred
    global max_accuracy
    global best_model
    global temp_arr
    accuracies = []
    cnn = create_lenet().to(device)
    cec = nn.L1Loss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    tensorprev_board = torch.tensor(prev_board, dtype=torch.float,requires_grad=True)
    tensorboard = torch.tensor(board, dtype=torch.float,requires_grad=True)
    optimizer.zero_grad()
    b = tensorprev_board.detach().numpy().reshape(int(WIDTH/size),int(HEIGHT/size))
    
    for i in range(0,len(b)):
        
        for j in range(0,len(b[i])):
            g = []
            h = []
            try:
                g.append(b[i-1][j-1])
                h.append(b[i-1][j-1])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i][j-1])
                h.append(b[i][j-1])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i+1][j-1])
                h.append(b[i+1][j-1])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i-1][j])
                h.append(b[i-1][j])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i+1][j])
                h.append(b[i+1][j])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i][j+1])
                h.append(b[i][j+1])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i+1][j+1])
                h.append(b[i+1][j+1])
            except:
                g.append(0)
                h.append(0)
            try:
                g.append(b[i-1][j+1])
                h.append(b[i-1][j+1])
            except:
                g.append(0)
                h.append(0)
            
            g = torch.tensor(g, dtype=torch.float,requires_grad=True)
            #print(g)
            pred = cnn(g)
            #print(pred)
        # pred = torch.tensor(temp_arr, dtype=torch.float,requires_grad=True)
            if pred.detach().numpy() > 0:
                num = 1.0
            else:
                num = 0.0
            k = tensorboard.detach().numpy().reshape(int(WIDTH/size),int(HEIGHT/size))
            #print(k[i][j])
            lab = torch.tensor(k[i][j], dtype=torch.float,requires_grad=True)
            number = torch.tensor(num, dtype=torch.float,requires_grad=True)
            loss = cec(number, lab)
            loss.backward()
            optimizer.step()
            out_pred.append(num)
            #out_pred = torch.tensor(out_pred, dtype=torch.float,requires_grad=True)
            #accuracy = float(validate(cnn, out_pred))
            #accuracies.append(accuracy)
            #out_pred = []   
   # if accuracy > max_accuracy:
       # best_model = copy.deepcopy(cnn)
       # max_accuracy = accuracy
        #print("Saving Best Model with Accuracy: ", accuracy)
        
   # print('Epoch:', epoch, "Accuracy :", accuracy, '%')
   # plt.plot(accuracies)


def draw_pred():
    global out_pred
    print(out_pred)
    temp = np.array(out_pred).reshape(int(WIDTH/size),int(HEIGHT/size))
    for i in range(0,len(temp)):
        
        for j in range(0,len(temp[i])):
            if temp[i][j] > 0:
                
                color = (3,176,26)
            else:
                
                color = (3,0,26)
                #row.append(0)
            pygame.draw.rect(WIN, color, pygame.Rect(i*size, j*size, i+size, j+size))#draws the squres
    out_pred = []
counter = 0

while True:#pygame Loop
    counter+=1
    next_gen()
    train(epoch=counter)
    draw_pred()
    
    time.sleep(0.05)
    pygame.display.update()
    for event in pygame.event.get():
        
        if event.type == pygame.QUIT:
            
            exit()

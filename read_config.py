from models import SLNN, MLNN, get_pre_activation_slnn, get_pre_activation_mlnn 

import numpy as np
from sys import argv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colormaps

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from copy import deepcopy

from logging import Logger
logger = Logger(__name__)

if __name__ == '__main__':
    if len(argv) != 6:
        print("launch with argv: L, num_conf, num_T, num_epochs, model_name")
        assert len(argv) == 6 

    L = int(argv[1])
    num_conf = int(argv[2])
    num_T = int(argv[3])
    num_epochs = int(argv[4])
    model_name = str(argv[5])
    explore = False

     
        
     

    fname = f"configs_{L}x{L}.txt"
    data = np.loadtxt(fname, delimiter=' ')

    T_low, T_high = 0.05, 5. 
    assert data.shape == (num_conf * num_T, L * L), data.shape
    

    
    T_range = np.linspace(T_low, T_high, num_T-1)

    X = data
    if explore:
        x_backup = deepcopy(X)
        X = X[np.sum(X, axis=1) >= 0]
        M = X.T @ X
        pca = PCA(n_components=10) 
        x_t = pca.fit(M) 
        e_v = pca.explained_variance_
        s = np.sum(e_v)

        plt.scatter(range(len(e_v)), e_v / s)
        plt.show()

        lam, eigenv = np.linalg.eig(M)
        

        y_ls = []
        for l in range(4):
            #y_l = X.dot(eigenv[:, l]) # -- something's off here
            y_l = X.dot(pca.components_[l])
            y_ls.append(y_l)

        fig, axs = plt.subplots(nrows=4, ncols=4)
        
        cmap = colormaps['RdYlBu']
        
        colors = cmap(-(T_range - T_high) / (T_high - T_low))
        mult_fact = 1. #10. -- 10 as maybe in original paper?
        for li in range(4):
            for lj in range(4):
                if li <= lj: continue
                for ti, t in enumerate(T_range):
                    axs[li, lj].scatter(
                            mult_fact * y_ls[li][ti * num_conf : (ti + 1) * num_conf],
                            mult_fact * y_ls[lj][ti * num_conf : (ti + 1) * num_conf],
                            color=colors[ti], s=.5) 
        
        plt.show()



        ts = np.linspace(T_low, T_high, num_T)
        ps, psigma = [], []
        for ti, t in enumerate(ts):
            s = []
            for n in range(num_conf):
                s.append(np.abs(np.sum(data[ti * num_conf + n]) / L ** 2))
            ps.append(np.mean(s))
            psigma.append(np.std(s))

        ps, psigma = np.array(ps), np.array(psigma)    
        plt.plot(ts, ps, color="red")
        plt.scatter(ts, ps + psigma, color="grey", alpha=.7, linestyle='-.')
        plt.scatter(ts, ps - psigma, color="grey", alpha=.7, linestyle='-.')
        plt.xlabel("T")
        plt.ylabel("<m>")
        plt.vlines(2.269, ymin=0, ymax=1, color="yellow")
        plt.show()

        fig, axs = plt.subplots(ncols=2, nrows=2)
        for i in range(2):
            axs[0][i].imshow(data[2 * num_conf + i].reshape((L, L)))
        for i in range(2):
            axs[1][i].imshow(data[-1 - i].reshape((L, L)))
        plt.show()
        X = x_backup



    def train_model(model, X, y, model_name, epochs=50, learning_rate=1e-2, batch_size=16,):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
         
        rand_idx = torch.randperm(y.size()[0])
        X_tensor = X[rand_idx] 
        y_tensor = y[rand_idx] 
        
        if model_name == "slnn":
            exclude_idx = torch.sum(X_tensor / L ** 2, axis=1) >= 0         
            X_tensor = X_tensor[exclude_idx][(int(X_tensor[exclude_idx].shape[0]) % batch_size):]
            y_tensor = y_tensor[exclude_idx][(int(y_tensor[exclude_idx].shape[0]) % batch_size):]
  
  
        for epoch in tqdm(range(epochs)):     
            for i in range(0, X_tensor.shape[0], batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    slnn = SLNN(input_size=L ** 2)
    mlnn = MLNN(input_size=L ** 2, hidden_size=2)
    if model_name == "slnn":
        model = slnn
    else:
        model = mlnn
    
 
    T_c = 2.269

    ferro_lo, ferro_hi = 0.05, 1.
    para_lo,  para_hi  = 4., 5.

    train_data_ferro = []
    train_data_para  = []
    test_data = []

    acceptable_restrict = [] 
    for i, t in enumerate(T_range):
        idx = np.sum(X[i * num_conf : (i + 1) * num_conf, :], axis=1) >= 0
        acceptable_restrict.append(idx)

        if t < ferro_hi:
            train_data_ferro.append(X[i * num_conf : (i + 1) * num_conf, :])
        elif t > para_lo:
            train_data_para.append(X[i * num_conf : (i + 1) * num_conf, :])
        else:
            test_data.append(X[i * num_conf : (i + 1) * num_conf, :])
            
    train_data_ferro = torch.tensor(np.array(train_data_ferro), dtype=torch.float32).reshape(-1, L ** 2)
    train_data_para  = torch.tensor(np.array(train_data_para),  dtype=torch.float32).reshape(-1, L ** 2)
    test_data        = torch.tensor(np.array(test_data),        dtype=torch.float32).reshape(-1, L ** 2)

      
    y_ferro = torch.ones((train_data_ferro.shape[0]))
    y_para  = 0 * torch.ones((train_data_para.shape[0]))

    train_data  = torch.cat((train_data_ferro, train_data_para)) 
    train_label = torch.cat((y_ferro, y_para)) 
    train_model(model, train_data, train_label, model_name, num_epochs)
   

    idx = np.sum(X, axis=1) >= 0
    if model_name == "hlnn":
        test_outputs = model(torch.tensor(X, dtype=torch.float32))
    else:
        test_outputs = model(torch.tensor(X[idx], dtype=torch.float32))
    
    test_outputs = test_outputs.detach().numpy() 
    plt.scatter(np.linspace(0, test_outputs.shape[0], test_outputs.shape[0]), test_outputs)
    plt.show()

    plt.cla()

    


    if model_name == "slnn":
        idx = np.sum(X, axis=1) >= 0
        args = get_pre_activation_slnn(model, torch.tensor(X[idx], dtype=torch.float32))
        w = model.linear.weight.detach().numpy().flatten()
        b = model.linear.bias.item()
        w_avg = np.mean(w)
        
        z1 = (X[idx] @ w)
        z2 = (X[idx] @ w + b) 
        
        plt.scatter(range(len(z1)), z1)
        plt.scatter(range(len(z2)), z2)
        plt.show()

        
        x = X @ w / np.max(X @ w)
        y = np.sum(X, axis=1) / L ** 2
        plt.scatter(x, y)
        plt.show()
         


    else:
        args1, args3 = get_pre_activation_mlnn(model, torch.tensor(X, dtype=torch.float32))
        mags = np.sum(X, axis=1) / L ** 2
        args3 = model.sigmoid(args3)
    
        plt.scatter(mags, args3[:, 0].detach().numpy(), label="final")
        plt.scatter(mags, args1[:, 0].detach().numpy(), label="u1")
        plt.scatter(mags, args1[:, 1].detach().numpy(), label="u2")
        plt.legend()
        plt.show()
        

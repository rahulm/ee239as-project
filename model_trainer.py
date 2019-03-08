import torch
import torch.nn as nn
import torch.optim as optim
import datetime 
import matplotlib.pyplot as plt

class ae_trainer(object):
    def __init__(self, use_cuda, model, loss_func, optimizer, model_name):
        self.model = model
        self.model_name = model_name
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        self.loss_func = loss_func
        self.optim = optimizer
        
    def train_model(self, epochs, trainloader):
        self.model.train()
        print("Beginning to train {} model".format(self.model_name))
        num_epoch = [0 for i in range(epochs)]
        train_loss = []

        for epoch in range(epochs):
            training_loss = 0
            for batch_num, batch in enumerate(trainloader):
                if self.use_cuda:
                    batch = batch.cuda()
                self.optim.zero_grad()
                x_recon = self.model(batch)
                loss = self.loss_func(x_recon, batch)
                loss.backward()
                self.optim.step()
                training_loss += loss.item()

            train_loss.append(training_loss)
            print('{} Model training epoch {}, Loss: {:.6f}'.format(self.model_name, epoch, training_loss/len(trainloader)))
        
        torch.save(self.model.state_dict, "./saved_weights/{}_model_weights".format(self.model_name) + datetime.datetime.now().strftime("%d%B%Y-%H_%M") + ".pth")

        plt.plot(num_epoch, train_loss)
        plt.ylabel("Training loss")
        plt.xlabel("Number of Epochs")
        plt.title("{} Model Training Loss vs Number of Epochs".format(self.model_name))
        plt.savefig('./train_loss_plots/{}_model_train_loss'.format(self.model_name) + datetime.datetime.now().strftime("%d%B%Y-%H_%M") + '.png')

class vae_trainer(object):
    def __init__(self, use_cuda, model, recon_loss_func, optimizer, model_name):
        self.model = model
        self.model_name = model_name
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        self.recon_loss_func = recon_loss_func
        self.optim = optimizer
    
    def vae_loss(self, x, x_recon, mu, var, recon_loss_func):
        recon_loss = recon_loss_func(x_recon, x)

        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return recon_loss + KLD

    def train_model(self, epochs, trainloader):
        self.model.train()
        print("Beginning to train {} model".format(self.model_name))
        num_epoch = [0 for i in range(epochs)]
        train_loss = []

        for epoch in range(epochs):
            training_loss = 0
            for batch_num, batch in enumerate(trainloader):
                if self.use_cuda:
                    batch = batch.cuda()
                self.optim.zero_grad()
                x_recon, mu, var = self.model(batch)
                loss = self.vae_loss(x_recon, batch, mu, var, self.recon_loss_func)
                loss.backward()
                self.optim.step()
                training_loss += loss.item()
                print("Batch {} done".format(batch_num))

            train_loss.append(training_loss)
            print('{} Model training epoch {}, Loss: {:.6f}'.format(self.model_name, epoch, training_loss/len(trainloader)))
        
        torch.save(self.model.state_dict, "./saved_weights/{}_model_weights".format(self.model_name) + datetime.datetime.now().strftime("%d%B%Y-%H_%M") + ".pth")

        plt.plot(num_epoch, train_loss)
        plt.ylabel("Training loss")
        plt.xlabel("Number of Epochs")
        plt.title("{} Model Training Loss vs Number of Epochs".format(self.model_name))
        plt.savefig('./train_loss_plots/{}_model_train_loss'.format(self.model_name) + datetime.datetime.now().strftime("%d%B%Y-%H_%M") + '.png')




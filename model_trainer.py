import torch
import torch.nn as nn
import torch.optim as optim
import datetime 
import matplotlib.pyplot as plt
import os
import csv

class ae_trainer(object):
    def __init__(self, use_cuda, model, loss_func, optimizer, model_name, exp_config):
        self.model = model
        self.model_name = model_name
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.loss_func = loss_func
        self.optim = optimizer
        self.exp_config = exp_config
        
    def train_model(self, epochs, trainloader, valloader):
        self.model.train()
        print("Beginning to train {} model".format(self.model_name))
        train_loss = []
        val_loss = []
        
        # setup csv for train loss
        train_loss_csv = open(os.path.join(self.exp_config.exp_metrics_dir, "train_loss.csv"), 'w', newline='')
        train_csv_writer = csv.writer(train_loss_csv)
        train_csv_writer.writerow(["epoch", "train_loss"])
        train_loss_csv.flush()
        
        # setup csv for val loss
        val_loss_csv = open(os.path.join(self.exp_config.exp_metrics_dir, "val_loss.csv"), 'w', newline='')
        val_csv_writer = csv.writer(val_loss_csv)
        val_csv_writer.writerow(["epoch", "val_loss"])
        val_loss_csv.flush()
        
        # setup csv for train loss
        train_loss_csv = open(os.path.join(self.exp_config.exp_metrics_dir, "train_loss.csv"), 'w', newline='')
        csv_writer = csv.writer(train_loss_csv)
        csv_writer.writerow(["epoch", "train_loss"])
        train_loss_csv.flush()

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

            training_loss_norm = training_loss/len(trainloader)
            train_loss.append(training_loss_norm)
            print('{} Model training epoch {}, Loss: {:.6f}'.format(self.model_name, epoch, training_loss_norm))
            
            # save model checkpoint
            torch.save(self.model.state_dict, os.path.join(self.exp_config.exp_models_dir, "{}-weights-epoch_{}.pth".format(self.model_name, epoch)))
            
            # save training loss
            train_csv_writer.writerow([str(epoch), "{:f}".format(training_loss_norm)])
            train_loss_csv.flush()
            
            
            # val phase
            self.model.eval()
            validation_loss = 0
            for batch_num, batch in enumerate(valloader):
                if self.use_cuda:
                    batch = batch.cuda()
                else:
                    batch = batch.cpu()
                x_recon, mu, var = self.model(batch)
                loss = self.vae_loss(batch, x_recon, mu, var, self.recon_loss_func)
                validation_loss += loss.item()
                print("Batch {} done".format(batch_num))
            
            validation_loss_norm = validation_loss/len(valloader)
            val_loss.append(validation_loss_norm)
            print('{} Model validation epoch {}, Loss: {:.6f}'.format(self.model_name, epoch, validation_loss_norm))
            
            # save validation loss
            val_csv_writer.writerow([str(epoch), "{:f}".format(validation_loss_norm)])
            val_loss_csv.flush()
            
        
        curr_date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
        
        plt.figure()
        plt.plot(range(epochs), train_loss, label='train')
        plt.plot(range(epochs), val_loss, label='val')
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Number of Epochs")
        plt.title("{} Model Loss vs Number of Epochs".format(self.model_name))
        plt.savefig(os.path.join(self.exp_config.exp_loss_plots_dir, '{}-loss-{}.png'.format(self.model_name, curr_date_time)))
        
        # flush and close loss
        train_loss_csv.flush()
        train_loss_csv.close()
        val_loss_csv.flush()
        val_loss_csv.close()
        
        

class vae_trainer(object):
    def __init__(self, use_cuda, model, recon_loss_func, optimizer, model_name, exp_config):
        self.model = model
        self.model_name = model_name
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.recon_loss_func = recon_loss_func
        self.optim = optimizer
        self.exp_config = exp_config
    
    def vae_loss(self, x, x_recon, mu, var, recon_loss_func):
        recon_loss = recon_loss_func(x_recon, x)

        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return recon_loss + KLD

    def train_model(self, epochs, trainloader, valloader):
        # self.model.train()
        print("Beginning to train {} model".format(self.model_name))
        train_loss = []
        val_loss = []
        
        # setup csv for train loss
        train_loss_csv = open(os.path.join(self.exp_config.exp_metrics_dir, "train_loss.csv"), 'w', newline='')
        train_csv_writer = csv.writer(train_loss_csv)
        train_csv_writer.writerow(["epoch", "train_loss"])
        train_loss_csv.flush()
        
        # setup csv for val loss
        val_loss_csv = open(os.path.join(self.exp_config.exp_metrics_dir, "val_loss.csv"), 'w', newline='')
        val_csv_writer = csv.writer(val_loss_csv)
        val_csv_writer.writerow(["epoch", "val_loss"])
        val_loss_csv.flush()

        for epoch in range(epochs):
            
            # train phase
            self.model.train()
            training_loss = 0
            for batch_num, batch in enumerate(trainloader):
                if self.use_cuda:
                    batch = batch.cuda()
                else:
                    batch = batch.cpu()
                self.optim.zero_grad()
                x_recon, mu, var = self.model(batch)
                loss = self.vae_loss(batch, x_recon, mu, var, self.recon_loss_func)
                loss.backward()
                self.optim.step()
                training_loss += loss.item()
                print("Batch {} done".format(batch_num))
            
            training_loss_norm = training_loss/len(trainloader)
            train_loss.append(training_loss_norm)
            print('{} Model training epoch {}, Loss: {:.6f}'.format(self.model_name, epoch, training_loss_norm))
            
            # save model checkpoint
            torch.save(self.model.state_dict, os.path.join(self.exp_config.exp_models_dir, "{}-weights-epoch_{}.pth".format(self.model_name, epoch)))
            
            # save training loss
            train_csv_writer.writerow([str(epoch), "{:f}".format(training_loss_norm)])
            train_loss_csv.flush()
            
            
            # val phase
            self.model.eval()
            validation_loss = 0
            for batch_num, batch in enumerate(valloader):
                if self.use_cuda:
                    batch = batch.cuda()
                else:
                    batch = batch.cpu()
                x_recon, mu, var = self.model(batch)
                loss = self.vae_loss(batch, x_recon, mu, var, self.recon_loss_func)
                validation_loss += loss.item()
                print("Batch {} done".format(batch_num))
            
            validation_loss_norm = validation_loss/len(valloader)
            val_loss.append(validation_loss_norm)
            print('{} Model validation epoch {}, Loss: {:.6f}'.format(self.model_name, epoch, validation_loss_norm))
            
            # save validation loss
            val_csv_writer.writerow([str(epoch), "{:f}".format(validation_loss_norm)])
            val_loss_csv.flush()
            
        
        curr_date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
        
        plt.figure()
        plt.plot(range(epochs), train_loss, label='train')
        plt.plot(range(epochs), val_loss, label='val')
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Number of Epochs")
        plt.title("{} Model Loss vs Number of Epochs".format(self.model_name))
        plt.savefig(os.path.join(self.exp_config.exp_loss_plots_dir, '{}-loss-{}.png'.format(self.model_name, curr_date_time)))
        
        # flush and close loss
        train_loss_csv.flush()
        train_loss_csv.close()
        val_loss_csv.flush()
        val_loss_csv.close()


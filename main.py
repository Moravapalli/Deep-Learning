from ast import parse
import os
import torch.nn as nn


from unicodedata import decimal
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from Autoencoder import Model
import LPIPS
from utils import Custom_dataset
from torchvision import datasets, transforms
import logging
import datetime

import wandb
wandb.login()
time = datetime.datetime.now().strftime("%d-%m-%H-%M")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class Trainer:
    def __init__(self,args):
        self.model = Model(args)
        
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_ae = self.optimizers(args)       
        self.create_training()
        self.train(args)
    

    def optimizers(self,args):
        lr = args.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        return opt_ae

    @staticmethod
    def create_training():
        os.makedirs("outputs/results", exist_ok=True)
        os.makedirs("outputs/checkpoints", exist_ok=True)

    def train(self,args):
        if args.model == 'CIFAR10':
            transform = transforms.ToTensor()
            # Download the training and test datasets
            train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, num_workers=0)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, num_workers=0)
        else:
            #Prepare custom data loaders
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)
            transform = transforms.Compose([transforms.Resize(size=512),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
            train_data = Custom_dataset(img_dir=args.dataset_path,size = 0.25,transform=transform)
            train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=False)
        
        steps_per_epoch = len(train_loader)
        model = self.model
        if args.pretrained == True:
            print('loading checkpoints............')
            #import pdb;pdb.set_trace()
            checkpoint = torch.load(args.checkpoints)
            pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
            model.load_state_dict(pretrained_dict)
            pre_epoch = checkpoint['epoch']
            print(f'starting from epoch_{pre_epoch}')
        else:
            pre_epoch = 0
        #model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()  # For training on distrubuted gpus
        model = model.cuda()
        for epoch in range(pre_epoch,pre_epoch+args.epochs):
            with tqdm(range(len(train_loader))) as pbar:
                for i, img in zip(pbar,train_loader):
                    img = img.float()
                    img = img.to(device=args.device)
                    decoder_img= model(img)
                    perceptual_loss = self.perceptual_loss(img, decoder_img)
                    
                    #reconstruction loss
                    rec_loss = torch.abs(img-decoder_img)
                    #LPIPS (vgg19)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    t_loss = perceptual_rec_loss + rec_loss
                    
                    self.opt_ae.zero_grad()
                    t_loss.sum().backward(retain_graph=True)

                    self.opt_ae.step()
    

                    checkpoints = {
                        "epoch":epoch,
                        "state_dict":model.state_dict(),
                    }
                        
                    wandb.log({'T_loss':t_loss.mean(),'rec_loss':rec_loss.mean(),'per_loss':perceptual_loss.mean()})
                    pbar.set_postfix(
                        Epoch_no = epoch,
                        T_loss=np.round(t_loss.mean().cpu().detach().numpy().item(), 5),
                        per_loss=np.round(perceptual_loss.cpu().detach().numpy().item(), 3),
                        R_Loss = np.round(rec_loss.mean().cpu().detach().numpy().item(),3)
                    )
                
                    pbar.update(0)

                    if epoch % 100 == 0 and i % 500 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((img[:4], decoder_img[:4]))  # add(1).mul(0.5)
                            vutils.save_image(real_fake_images,os.path.join("outputs/results", f"{epoch}_{i}.jpg"),
                                              nrow=4)

                            torch.save(checkpoints, os.path.join("outputs/checkpoints", f"ckpt.pt"))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Autoencoder with Lpips")
    parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-06, help='Learning rate (default: 0.0002)') #0.0000
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--pretrained',type=bool,default=True,help='resume training from previous checkpoints')
    parser.add_argument('--checkpoints',type=str,default='/checkpoints',help='load checkpoints for resume the training or inference')
    parser.add_argument('--model', type=str, default='custom', help='For saving the log for each model')
    parser.add_argument('--enc_channels', type=str, default=[(3,8)], help='encoder channels')
    parser.add_argument('--dec_channels', type=str, default=[(512,256)], help='encoder channels')
    args = parser.parse_args()
    args.dataset_path = r'/home/moravapa/Documents/Thesis/data/train'
    args.checkpoints = r'/home/moravapa/checkpoints/resume/ckpt.pt'
    args.enc_channels = [(3,8), (8,16), (16,32), (32,64),(64, 128), (128,128), (128,256), (256,512)]
    args.dec_channels = [(512,256),(256,128),(128,128),(128,64),(64,32),(32,16),(16,8),(8,3)]
    base_dir = r'/home/moravapa/Documents/Thesis/outputs'
    dir = args.model + "_" + str(args.num_codebook_vectors) + "_" + str(args.latent_dim) + "_" + time
    # path = os.path.join(base_dir,dir)
    path = os.path.join(base_dir, dir)
    if not os.path.isdir(path):
        os.mkdir(path)

    wandb.init(project="run_1", name="project", dir=path)
    wandb.config.update(args)
    config = wandb.config
    wandb.run.name = dir
    train_vggan = Trainer(args) 

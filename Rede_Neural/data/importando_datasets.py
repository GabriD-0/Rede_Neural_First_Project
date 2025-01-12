import torch
from torchvision import datasets, transforms

# Definindo a conversão de imagem para tensor
transforms = transforms.ToTensor() 

trainset = datasets.MNIST(
    root='./MNIST_data', 
    train=True, 
    download=True, 
    transform=transforms
    ) # Carregando o dataset

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=64, 
    shuffle=True
    ) # Criando um buffer para pegar os dados por partes


valset = datasets.MNIST(
    root='./MNIST_data', 
    train=False, 
    download=True, 
    transform=transforms
    ) # Carrega a parte de validação do dataset

valloader = torch.utils.data.DataLoader(
    valset, 
    batch_size=64, 
    shuffle=True
    ) # Cria um buffer para pegar os dados de validação por partes


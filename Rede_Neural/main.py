import torch
import matplotlib.pyplot as plt
from models.modelo import Modelo
from train import treino, validacao
from data.importando_datasets import trainloader, valloader

def main():
    # Definindo o uso de GPU (CUDA) ou CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo em uso:", device)

    # Definindo a rede, instancia o modelo e move para device
    modelo = Modelo().to(device)

    # Treino
    treino(modelo, trainloader, device, epochs=7, lr=0.01, momentum=0.5)

    # Validacao
    validacao(modelo, valloader, device)
    
    # Exemplo de visualizacao de batch
    dataiter = iter(trainloader)
    imagens, etiquetas = next(dataiter)
    
    # Mostra a primeira imagem do batch
    plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
    plt.title(f"Etiqueta: {etiquetas[0].item()}")
    plt.show()
    
    # Mostrando shape (opcional)
    print("Shape da imagem [0]:", imagens[0].shape)
    print("Shape da etiqueta [0]:", etiquetas[0].shape)

if __name__ == "__main__":
    main()
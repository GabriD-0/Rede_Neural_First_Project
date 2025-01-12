import torch
import torch.nn as nn
import torch.nn.functional as F

# Definindo a rede
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        
        # Ajuste de nomes para manter consistência entre __init__ e forward
        self.linear1 = nn.Linear(28*28, 128)      # Primeira camada ou camada de entrada, 784 inputs(neuronios), 128 outputs
        self.linear2 = nn.Linear(128, 64)         # Camada interna 1, Segunda camada, 128 inputs, 64 outputs
        self.linear3 = nn.Linear(64, 10)          # Camada interna 2, Terceira camada, 64 inputs, 10 outputs
        # Para a camada de saida não precisa de um linear, pois ela vem direto da camada de saida(output da camada Interna 2)

    def forward(self, x):
        
        # x = x.view(x.shape[0], -1)        # Convertendo as imagens para "vetores" de 28*28 casas para ficarem compativeis com a rede
        x = F.relu(self.linear1(x))         # Função de ativação da camada de entrada para a camada interna 1
        x = F.relu(self.linear2(x))         # Função de ativação da camada interna 1 para a camada interna 2
        x = self.linear3(x)                 # Função de ativação da camada interna 2 para a camada de saida, nesse caso f(x) = x
    
        return F.log_softmax(x, dim=1)      # Dados utilizados para calcular a perda


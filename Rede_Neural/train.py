import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Otimização da Rede, definindo o algoritmo de otimização e a função de perda / pesos
def treino(modelo, trainloader, device, epochs = 7, lr = 0.01, momentum = 0.5):
    
    # Definindo a politica de atualização dos pesos e das bias
    otimizador = optim.SGD(modelo.parameters(), lr=lr, momentum=momentum) 
    criterio = nn.NLLLoss() # Definindo o criterio para calcular a perda
    
    modelo.train()
    inicio = time.time() # Contagem de tempo de treinamento
    #epochs = 10 # Definindo o numero de epochs(épocas) que o algoritmo ira rodar
    
    for epoch in range(epochs):
        # Definindo a perda acumulada        
        perda_acumulada = 0 
        
        for imagens, etiquetas in trainloader:
            # Convertendo as imagens para "vetores" de 28*28 casas para ficarem compativeis com a rede
            imagens = imagens.view(imagens.shape[0], -1).to(device) 
            etiquetas = etiquetas.to(device)
            
            otimizador.zero_grad() # Zerando os gradientes
            
            output = modelo(imagens) # Colocando as imagens(dados) na rede
            
            perda_instantanea = criterio(output, etiquetas) # Calculando a perda do epoch em questão
            
            perda_instantanea.backward() # Back propagation a partira da perda
            
            otimizador.step() # Atualizando os pesos e bias
            
            perda_acumulada += perda_instantanea.item() # Acumulando a perda
            
        print(f"Epoch [{epoch+1}/{epochs}] - Perda acumulada: {perda_acumulada:.4f}")
    
    fim = time.time()
    print(f"Tempo de treinamento: {fim - inicio:.2f} segundos")


# Validando o modelo
def validacao(modelo, valloader, device):
    
    modelo.eval() # Ativando o modo de validação (desativa dropout, etc.)
    conta_correta, conta_todas = 0, 0
    
    with torch.no_grad(): # sem cálculo de gradiente na validação
        for imagens, etiquetas in valloader:
            imagens = imagens.view(imagens.shape[0], -1).to(device) # Convertendo as imagens para "vetores" de 28*28 casas para ficarem compativeis com a rede
            etiquetas = etiquetas.to(device)            
            
            logps = modelo(imagens)
            ps = torch.exp(logps) # convertendo a saida para escala de probabilidade(tensor)
            
            # ps.shape = (batch_size, 10). Pegar a classe com maior probabilidade
            top_p, top_class = ps.topk(1, dim=1)
            
            # Conta quantas imagens foram classificadas corretamente
            equals = top_class.view(*etiquetas.shape).eq(etiquetas)
            conta_correta += equals.sum().item()
            conta_todas += len(etiquetas)
                    
    print("Total de imagens = ", conta_todas)        
    print(f"Precisão do Modelo: {conta_correta * 100.0 / conta_todas:.2f}%")
    

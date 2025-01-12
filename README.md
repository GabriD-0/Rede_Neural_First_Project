# Projeto: Rede Neural em PyTorch para Classificação de Dígitos MNIST

Este projeto implementa uma rede neural simples em PyTorch para classificar dígitos do dataset MNIST, usando funções de treinamento, validação e visualização de resultados.

---

## Sumário

- [Estrutura de Pastas](#estrutura-de-pastas)
- [Instalação e Configuração do Ambiente](#instalação-e-configuração-do-ambiente)
- [Como Executar](#como-executar)
- [Uso de GPU (CUDA)](#uso-de-gpu-cuda)
- [Descrição dos Arquivos](#descrição-dos-arquivos)
- [Resultados](#resultados)
- [Referências](#referências)

---

## Estrutura de Pastas

A estrutura sugerida para o projeto é:

```
Rede_Neural/
├── data/
│   ├── __init__.py
│   └── importando_datasets.py
├── models/
│   ├── __init__.py
│   └── modelo.py
├── Rede_Neural/
│   ├── main.py
│   ├── train.py
│   └── ...
├── .venv/                  # (opcional) Ambiente virtual Python
├── README.md               # Este arquivo
└── requirements.txt        # (opcional) Dependências
```
---

## Instalação e Configuração do Ambiente

1. **Clonar o repositório (opcional)**  
   ```bash
   git clone https://github.com/usuario/ML_First_Project.git
   cd Rede_Neural
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado)**  
   ```bash
   python -m venv .venv
   ```
   e ative-o:  
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

3. **Instale as dependências**  
   Se tiver um arquivo `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Ou instale manualmente:
   ```bash
   pip install torch torchvision matplotlib
   ```

---

## Como Executar

1. Abra um terminal dentro da pasta raiz do projeto (ex: `Rede_Neural/`).
2. Certifique-se de que seu ambiente virtual está ativo.
3. Rode o script principal:
   ```bash
   python Rede_Neural/main.py
   ```
   O código irá:
   - Baixar o dataset MNIST (caso não exista).
   - Treinar a rede neural por um número de épocas (definido no código).
   - Validar o modelo após o treino.
   - Exibir uma imagem de exemplo do batch de treinamento.

Você deverá ver algo como:

```
Dispositivo em uso: cuda
Epoch [1/10] - Perda acumulada: ...
...
Total de imagens = 10000
Precisão do Modelo: 95.76%
```

---

## Uso de GPU (CUDA)

1. **Verifique se você possui drivers atualizados da NVIDIA**.
2. **Instale a versão do PyTorch com suporte a CUDA**. Exemplo (PyTorch mais recente + CUDA 11.8):
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. No código, usamos:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```
   Se `torch.cuda.is_available()` for `True`, o script usará sua GPU. Caso contrário, usará a CPU.

---

## Descrição dos Arquivos

- **`data/importando_datasets.py`**  
  - Carrega e transforma o dataset MNIST (transformação `ToTensor()`).  
  - Exporta `trainset`, `trainloader`, `valset`, `valloader` com `batch_size` e `shuffle` configurados.

- **`models/modelo.py`**  
  - Define a classe `Modelo` (herdando de `nn.Module`), com camadas lineares e o método `forward`.

- **`train.py`**  
  - Define funções de treinamento (`treino()`) e validação (`validacao()`).  
  - Gerencia otimização, cálculo de perda (`NLLLoss`), e exibe métricas como perda acumulada e acurácia.

- **`main.py`**  
  - Script principal que:
    1. Define `device` (cuda ou cpu).
    2. Instancia o modelo e move para `device`.
    3. Chama `treino()`.
    4. Chama `validacao()`.
    5. Exibe exemplos de imagens e dimensões de tensores.

---

## Resultados

- Ao final de 10 épocas (epochs), o modelo deve alcançar uma acurácia acima de **94%** no MNIST, podendo chegar a **97~98%** conforme hiperparâmetros ou ajustes no modelo.
- Tempos de treinamento podem variar dependendo do uso de CPU ou GPU.

---

## Referências

- [Documentação do PyTorch](https://pytorch.org/)
- [Dataset MNIST](http://yann.lecun.com/exdb/mnist/)  
- [Exemplo de Classificação MNIST em PyTorch (Tutorial Oficial)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — apesar de focar em CIFAR10, os conceitos são semelhantes.  

Sinta-se à vontade para contribuir com melhorias ou abrir issues neste projeto!

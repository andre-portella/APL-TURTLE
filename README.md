# APL-TURTLE
Este repositório apresenta um projeto de Iniciação Científica focado na minimização do esforço de rotulação em tarefas de Visão Computacional. A abordagem consiste na aplicação de Active Prompt Learning (APL) em associação ao framework TURTLE, otimizando a seleção de dados para o ajuste fino de modelos de aprendizado profundo.


### Execução
Para executar o código, alterar o parâmetro `DATA` em `scripts/alvlm/main.sh`. Depois, você pode executar o código a partir do seguinte comando:
```bash
sh scripts/alvlm/main.sh [DATASET] [MODEL] [AL METHOD] [SEED NUMBER] [MODE] [SELECAO]
```
- **DATASET** $\em$ [oxford_flowers, dtd, oxford_pets, caltech101, eurosat, fgvc_aircraft, cifar10_custom, stl10_custom]
- **MODEL** $\em$ [RN50, RN101, vit_b32, vit_b16]
- **AL METHOD** $\em$ [cbsq]
- **SEED**: inteiro 
- **MODE**: Description augmentation $\em$ [none]
- **SELECAO**: $\em$ [centroide, entropia, confianca, margem, margem_confianca]
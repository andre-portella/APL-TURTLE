# APL-TURTLE
Este repositório apresenta o projeto de Iniciação Científica focado na minimização do esforço de rotulação em tarefas de Visão Computacional. A abordagem principal consiste na aplicação de Active Prompt Learning (APL) em associação ao framework TURTLE, otimizando a seleção de dados para o ajuste fino de modelos de aprendizado profundo.


### Execução
Para executar o código, alterar o parâmetro `DATA` em `scripts/alvlm/main.sh`. Depois, você pode executar o código a partir do seguinte comando:
```bash
sh scripts/alvlm/main.sh [DATASET] [MODEL] [AL METHOD] [SEED NUMBER] [MODE] 
```
- **DATASET** $\in$ [oxford_flowers, dtd, oxford_pets, caltech101, eurosat, fgvc_aircraft, cifar10_custom, stl10_custom]
- **MODEL** $\in$ [RN50, RN101, vit_b32, vit_b16]
- **AL METHOD** $\in$ [cbsq]
- **SEED**: integer 
- **MODE**: This is for description augmentation $\in$ [none]
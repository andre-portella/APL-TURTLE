# APL-TURTLE
Este repositório apresenta um projeto de Iniciação Científica focado na minimização do esforço de rotulação em tarefas de Visão Computacional. A abordagem consiste na aplicação de Active Prompt Learning (APL) em associação ao framework TURTLE, otimizando a seleção de dados para o ajuste fino de modelos de aprendizado profundo.


### Execução
Para executar o código, alterar o parâmetro `DATA` em `APL/scripts/alvlm/main.sh`. Pode ser preciso ajustar o caminho do framework TURTLE em: `APL/trainers/active_learning/turtle_runner.py`

Com os ajustes realizados, pode-se executar o projeto, dentro do diretório `APL`, a partir do seguinte comando:
```bash
sh scripts/alvlm/main.sh [DATASET] [MODEL] [AL METHOD] [SEED NUMBER] [MODE] [SELECAO]
```
- **DATASET** $\in$ [oxford_flowers, dtd, oxford_pets, caltech101, eurosat, fgvc_aircraft, cifar10_custom, stl10_custom]
- **MODEL** $\in$ [RN50, RN101, vit_b32, vit_b16]
- **AL METHOD** $\in$ [cbsq]
- **SEED**: inteiro 
- **MODE**: Description augmentation $\in$ [none]
- **SELECAO**: $\in$ [centroide, entropia, confianca, margem, margem_confianca]
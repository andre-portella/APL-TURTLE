# Instalar os Datasets
Colocar todos os conjuntos de dados sob a mesma pasta (chamada $DATA) para facilitar o gerenciamento e seguir as instruções abaixo para organizá-los, evitando a necessidade de modificar o código-fonte. A estrutura de arquivos deve seguir o seguinte padrão:

```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– dtd/
```


Lista de datasets:

- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [Flowers102](#flowers102)
- [FGVCAircraft](#fgvcaircraft)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [CIFAR-10](#cifar10_custom)
- [STL-10](#cifar10_custom)

As instruções para preparar cada dataset estão detalhadas abaixo. Para garantir a reprodutibilidade e uma comparação justa em trabalhos futuros, fornecemos divisões (splits) fixas de treino/val/teste para todos os datasets. As divisões fixas são provenientes dos datasets originais ou foram criadas por nós.


### Caltech101
- Crie um diretório de nome `caltech-101/` dentro de `$DATA`.
- Baixe `101_ObjectCategories.tar.gz` de http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz e extraia o arquivo em `$DATA/caltech-101`.
- Baixe `split_zhou_Caltech101.json` desse [link](https://drive.google.com/file/d/1pfTLKC1MtHe84CHo-VeS1D8mNkdqtaB2/view?usp=drive_link) e coloque em `$DATA/caltech-101`. 

A estrutura do diretório deve ser assim:
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
- Crie um diretório de nome `oxford_pets/` dentro de `$DATA`.
- Baixe as imagens de https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Baixe as anotações de https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Baaixe `split_zhou_OxfordPets.json` desse [link](https://drive.google.com/file/d/1E59ZB1-tEJO0HXUSTZQzER4zt3ZrXGIH/view?usp=drive_link). 

A estrutura do diretório deve ser assim:
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### Flowers102
- Crie um diretório de nome `oxford_flowers/` dentro de `$DATA`.
- Baixe as imagens e rótulos de https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz e https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectivamente.
- Baixe `cat_to_name.json` de [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing). 
- Baixe `split_zhou_OxfordFlowers.json` desse [link](https://drive.google.com/file/d/118yF_xkSHZ_kQRjGCuH6SoU_lrtHqOHd/view?usp=drive_link).

A estrutura do diretório deve ser assim:
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### FGVCAircraft
- Baixe os dados de https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extraia `fgvc-aircraft-2013b.tar.gz` e mantenha somente `data/`.
- Mova `data/` para `$DATA` e renomeie o diretório para`fgvc_aircraft/`.

A estrutura do diretório deve ser assim:
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### DTD
- Baixe o dataset https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz e extraia para `$DATA`.
- Baixe `split_zhou_DescribableTextures.json` desse [link](https://drive.google.com/file/d/1IO0D-ZmEvlag56_u-SuQpMxx0ufh2zz-/view?usp=drive_link).

A estrutura do diretório deve ser assim:
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Crie um diretório de nome `eurosat/` dentro de `$DATA`.
- Baixe o dataset de https://zenodo.org/records/7711810 e extraia para`$DATA/eurosat/`.
- Baixe `split_zhou_EuroSAT.json` desse [link](https://drive.google.com/file/d/1YgFd15Ra1wz9p7PDz4aR6ra7Lp1g4AfQ/view?usp=drive_link).

A estrutura do diretório deve ser assim:
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### CIFAR-10
- Crie um diretório de nome `cifar10/` dentro de `$DATA`.
- Baixe o dataset de https://www.cs.toronto.edu/~kriz/cifar.html e extraia para`$DATA/cifar10/`. 
- Baixe `split_zhou_.json` desse [link](https://drive.google.com/file/d/1gLJyDsyt_A5bcxI8IU8wrH0XMyRKLKV_/view?usp=drive_link).

### STL-10
- Crie um diretório de nome `stl10/` dentro de `$DATA`.
- Baixe o dataset de https://cs.stanford.edu/~acoates/stl10/ e extraia para`$DATA/stl10/`. 
- Baixe `split_zhou_.json` desse [link](https://drive.google.com/file/d/1PdGx5KIwErP-MNJGtLxePJnAKN2cHSOE/view?usp=drive_link).
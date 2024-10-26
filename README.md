# Sistema de Tradução em Libras para Português

Este projeto é uma aplicação web que utiliza técnicas de visão computacional e aprendizado de máquina para traduzir gestos de Libras (Língua Brasileira de Sinais) em texto em português. O sistema foi desenvolvido em Python com a biblioteca Flask e usa MediaPipe para detecção de landmarks das mãos, com um modelo de Random Forest para classificação dos gestos.


## Funcionalidades Principais

### Captura de Gestos em Libras: Utiliza a câmera para capturar gestos em tempo real.
### Processamento de Imagens e Landmarks: Extrai landmarks das mãos usando MediaPipe.
### Classificação de Gestos com Random Forest: Treina um modelo de Random Forest para classificação dos gestos com base nos landmarks.
### Tradução em Tempo Real: Transmite a tradução do gesto capturado pela câmera diretamente na interface web.


## Tecnologias e Bibliotecas
### Python: Linguagem principal do backend.
### Flask: Framework para criar a API e servir a interface web.
### OpenCV: Utilizado para manipulação de imagens e captura de vídeo.
### MediaPipe: Biblioteca para detecção e rastreamento dos landmarks das mãos.
### Scikit-learn: Usado para treinar o modelo de Random Forest.
### Pandas: Manipulação de dados para carregar e processar o dataset de landmarks.
### NumPy: Suporte para cálculos e operações com arrays.
### CSV: Armazenamento dos dados de landmarks e rótulos em um arquivo CSV.


## Estrutura do Projeto
### app.py: Arquivo principal do projeto que contém todas as rotas e lógica do backend.
### images/: Pasta onde as imagens de treinamento para o modelo são armazenadas.
### libras_dataset.csv: Arquivo CSV onde são salvos os landmarks das mãos e os rótulos dos gestos.
### templates/: Contém os templates HTML, incluindo translate.html para o stream de vídeo em tempo real.
### requirements.txt: Lista de dependências do projeto.


## Instalação e Configuração


### Clone o repositório:


git clone https://github.com/gabrielgcosta/libras-translate
cd libras-translate

### Instale as dependências:
pip install -r requirements.txt

### Crie as pastas necessárias:
Certifique-se de que a pasta images existe na raiz do projeto.

### Execute o aplicativo:
python app.py

### Acesse o sistema:
Abra o navegador e vá para http://localhost:5000/translate para visualizar o stream de vídeo e testar a tradução em tempo real.

## Endpoints e API
## /process_images [POST]
Processa todas as imagens na pasta images, extrai os landmarks e salva no CSV para posterior treinamento do modelo.

Resposta de Sucesso: { "message": "Processed images: [lista de imagens]" }
Erro: { "error": "Images folder not found" }

## /train [POST]
Treina o modelo de Random Forest usando os dados salvos no CSV.

Resposta de Sucesso: { "message": "Model trained successfully!" }
Erro: { "error": "No data available to train the model" }
/translate [GET]
Renderiza a página translate.html que exibe o stream de vídeo para a tradução em tempo real dos gestos.

/video_feed [GET]
Fornece o stream de vídeo em tempo real para a página translate.html.

Organização do Código e Fluxo de Dados
Captura de Imagens: As imagens de treinamento são colocadas na pasta images. Os nomes dos arquivos indicam o rótulo do gesto (e.g., "saudacao-01.jpg").
Processamento e Extração de Landmarks: A função process_images carrega cada imagem, detecta os landmarks das mãos e salva no CSV.
Treinamento do Modelo: O endpoint /train utiliza o arquivo CSV para treinar um modelo Random Forest com os dados capturados.
Tradução em Tempo Real: A função generate_frames captura o vídeo da câmera em tempo real e detecta os landmarks das mãos. Estes landmarks são passados para a função predict_gesture, que usa o modelo treinado para prever o gesto e exibir a tradução na tela.
Estrutura do Dataset (CSV)
O arquivo libras_dataset.csv contém as seguintes colunas:

label: O rótulo do gesto.
landmark_{i}_{axis}: Coordenadas x, y e z dos 21 landmarks da mão. Isso resulta em 63 colunas adicionais para cada landmark.

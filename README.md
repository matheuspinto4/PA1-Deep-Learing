# PA1 - Deep Learning: Segmentação Semântica e de Instâncias

Trabalho da disciplina de Aprendizado Profundo (FGV) — segmentação de núcleos celulares
(dataset Data Science Bowl 2018) evoluindo de uma baseline binária ingênua até uma
cabeça de instância treinada de ponta a ponta.

## Ambiente

- Python 3.13 (testado)
- Instalar dependências:

```bash
pip install -r requirements.txt
```

- Roda em CPU (treinos de poucos minutos por época, imagens redimensionadas para 128x128).
  GPU é opcional e usada automaticamente se disponível (`torch.cuda.is_available()`).

## Dados

O projeto usa o dataset **Data Science Bowl 2018** (`stage1_train`). A pasta `data/` está
no `.gitignore` e não é versionada — baixe o dataset e extraia de forma que a estrutura
fique:

```
data/
└── stage1_train/
    └── <id_da_imagem>/
        ├── images/   (uma imagem .png)
        └── masks/    (uma máscara .png por núcleo, sem sobreposição)
```

## Estrutura do projeto

- `dataset.py` — `DSB2018Dataset` (dados reais) e `SyntheticEllipseDataset` (Parte 0)
- `model.py` — `ModularUNet`: encoder ResNet34 pré-treinado + decoder U-Net com skip connections
- `instance_labels.py` — gera o rótulo de 3 classes (fundo/interior/fronteira) a partir da máscara de instância
- `losses.py` — `DiceLoss` e `compute_class_weights` (peso por classe via frequência inversa)
- `metrics.py` — métricas semânticas (IoU/Dice, IoU por classe) e a métrica de instância (mAP @ [0.50:0.95] com matching guloso + erro de contagem) — fonte única usada por todas as partes
- `postprocessing.py` — extração de instâncias: `connected_components_to_instances` (Parte 1) e `watershed_to_instances` (Parte 2)
- `train_synthetic.py` / `train.py` / `train_instance_head.py` — scripts de treino de cada parte
- `evaluate.py` — avaliação de instância do baseline e da Trilha A, lado a lado
- `check_border_size.py` — diagnóstico da espessura de erosão usada na Parte 2

Checkpoints (`resultados/modelos/*.pth`) não são versionados (`.gitignore`); gráficos e
visualizações ficam em `resultados/imagens/`.

## Parte 0 — Teste unitário sintético

Antes de treinar no dataset real, `SyntheticEllipseDataset` gera imagens 128x128 com 5 a
20 elipses de tamanhos variados, algumas se tocando, com ruído gaussiano e contraste
aleatórios — e cada elipse tem sua própria máscara de instância. Serve como um teste
unitário rápido: se o `ModularUNet` não consegue nem segmentar elipses sintéticas, não
faz sentido gastar tempo no dataset real.

```bash
python train_synthetic.py
```

Treina 5 épocas (BCE + Dice Loss), imprime Loss/IoU/Dice de validação por época e salva
`resultados/imagens/synthetic_results.png` com uma comparação visual (entrada, gabarito,
previsão) de 3 amostras.

## Parte 1 — Baseline binário + extração ingênua de instâncias

O `ModularUNet` é treinado para segmentação **binária** (fundo vs. núcleo) no dataset
real, com BCE + Dice Loss. A partir da máscara binária prevista, instâncias são extraídas
da forma mais ingênua possível: limiar em 0.5 seguido de **componentes conexos**
(`scipy.ndimage.label`) — sem nenhuma lógica para separar núcleos que se tocam.

```bash
python train.py       # treina o modelo binário, salva resultados/modelos/best_baseline_model.pth
python evaluate.py    # avalia o baseline (e a Trilha A, se já treinada) no nível de instância
```

A avaliação usa a métrica de instância de `metrics.py`: mAP médio sobre os limiares de
IoU de 0.50 a 0.95 (passo 0.05), com **matching guloso** — para cada limiar, os pares
(predição, GT) com IoU acima do limiar são ordenados por IoU decrescente e casados sem
reaproveitar predição nem GT já casados — além do erro absoluto médio de contagem de
objetos por imagem.

Resultado observado: mAP médio de **0.0949** e erro de contagem médio de **29.82**
núcleos por imagem — o método ingênuo funde núcleos encostados em um só blob, então
conta muito menos objetos do que existem de verdade. Esse fracasso cresce com a
densidade de núcleos por imagem (ver `resultados/imagens/baseline_failure_analysis.png`).

## Parte 2 — Trilha A: fronteiras + watershed

Para separar núcleos encostados, o `ModularUNet` passa a prever **3 classes** por pixel
(fundo / interior do núcleo / fronteira entre núcleos vizinhos) em vez de só fundo/objeto,
e a extração de instância passa a ser feita por watershed guiado por marcadores.

**Como o rótulo de 3 classes é gerado** (`instance_labels.py`): cada instância da máscara
é erodida **individualmente** — o que a erosão consome vira fronteira, o que sobrevive
vira interior. Como a erosão de cada núcleo ignora os vizinhos, dois núcleos encostados
acabam com interiores separados por um "fosso" de fronteira entre eles, mesmo que as
máscaras originais se tocassem.

A espessura da erosão não é fixa — é um **teto adaptativo** de até 2px, limitado pelo
raio inscrito (via transformada de distância) de cada núcleo. Isso foi uma correção
sobre a primeira versão: com um teto fixo de 2px, **40.1%** dos núcleos do dataset
perdiam o interior por completo (núcleos pequenos, área média de só 13.7px²), o que
os deixava sem nenhum marcador no watershed. Com o teto adaptativo, esse número caiu
para **3.7%** (`check_border_size.py` reproduz essa medição).

A classe fronteira é sempre minoritária (é só um anel fino ao redor de cada núcleo),
então a loss é uma **Cross-Entropy balanceada** (`nn.CrossEntropyLoss` com peso por
classe), com pesos calculados por frequência inversa de pixels
(`losses.compute_class_weights`) — na prática, a fronteira recebe um peso ~4-5x maior
que o fundo.

Na inferência, o mapa de 3 classes é decodificado de volta em instâncias por
**watershed controlado por marcadores** (`postprocessing.watershed_to_instances`): cada
componente conexo da classe interior vira um marcador, a "água" de cada marcador sobe
seguindo a superfície `1 - prob(interior)` e para nos pixels classificados como fundo.

```bash
python instance_labels.py     # gera um exemplo do rotulo de 3 classes e uma visualizacao de sanidade
python check_border_size.py   # diagnostico: quantifica nucleos que perderiam o interior por completo
python train_instance_head.py # treina a cabeca de 3 classes, salva resultados/modelos/best_instance_head_model.pth
python evaluate.py            # avalia baseline e Trilha A juntos, imprime a tabela comparativa
```

Resultado observado (mesmo conjunto de validação da Parte 1):

| Método | mAP @ [0.50:0.95] | Erro de contagem médio |
|---|---|---|
| Componentes conexos (Parte 1) | 0.0949 | 29.82 |
| Fronteiras + watershed (Parte 2) | 0.3438 | 11.76 |

O ganho é maior justamente nas imagens mais densas: na amostra com 123 núcleos reais, o
mAP subiu de 0.0799 para 0.4828 e o erro de contagem caiu de 62 para 20 — ainda não é
perfeito (ver `resultados/imagens/instance_head_failure_analysis.png`), mas mostra que
separar núcleos encostados é exatamente onde o método ingênuo mais falha.

## Parte 3 — Ablações (Eixo 1: resolução, Eixo 2: função de perda)

Duas ablações rodadas com o mesmo modelo da Parte 2 (Trilha A), cada configuração com
2 seeds.

**Eixo 1 — mecanismo de recuperação de resolução.** Comparação entre o decoder atual
(`skip`, U-Net com skip connections) e uma segunda variante (`atrous_aspp`) que substitui
skip connections por convolução atrous mantendo o *output stride* em 8 (mesmo encoder
ResNet34, com `layer3`/`layer4` convertidos de stride para dilatação — o
`replace_stride_with_dilation` do torchvision não funciona no ResNet34 porque o
`BasicBlock` recusa `dilation > 1` no construtor, então os atributos de stride/dilatação
das convoluções já construídas são editados diretamente) seguida de uma ASPP
(convoluções atrous em paralelo com taxas 6/12/18 + ramo de image pooling, slide 57).

**Eixo 2 — função de perda.** Uma única classe (`FocalLoss(gamma, weight)`) generaliza os
4 tipos pedidos: `gamma=0` reduz-se exatamente à Cross-Entropy (com ou sem peso por
classe), `gamma>0` vira Focal (balanceada ou não). Variado `gamma ∈ {0,1,2,5}` × peso por
classe ∈ {sim, não}, decoder fixo em `skip`.

Devido ao prazo, as ablações usaram 3 épocas (em vez das 5 usadas na Parte 2) — o
objetivo é o efeito relativo entre configurações, não o melhor checkpoint absoluto.

```bash
python losses.py                # teste de sanidade: FocalLoss(gamma=0) == CrossEntropyLoss
python model.py                  # teste de sanidade: as duas variantes de decoder produzem a mesma forma de saida
python run_ablations.py          # roda as 18 configuracoes unicas (~1h), salva resultados/ablacoes_resultados.csv
python summarize_ablations.py    # agrega os resultados e gera resultados/imagens/eixo2_loss_ablation.png
```

### Resultados — Eixo 1 (gamma=0, balanceada, 2 seeds)

| Decoder | mAP | IoU fronteira | Erro de contagem |
|---|---|---|---|
| `skip` | 0.2475 ± 0.0512 | 0.4281 ± 0.0438 | 12.54 ± 0.01 |
| `atrous_aspp` | 0.0679 ± 0.0037 | 0.2467 ± 0.0172 | 36.69 ± 0.10 |

O `atrous_aspp` perdeu em todas as métricas, com desvio pequeno entre seeds (o efeito é
consistente, não ruído). Explicação mais provável: essa variante faz upsample bilinear
"burro" de 8x no final, sem nenhum mecanismo aprendido para recuperar detalhe fino — ao
contrário do `skip`, que reconstrói a resolução em 4 estágios aprendidos com acesso direto
às features de alta resolução do encoder via skip connections. É consistente com o motivo
pelo qual o DeepLabv3+ (slide 61) reintroduz skip connections em cima do ASPP puro do
DeepLabv3. **Ressalva**: com só 3 épocas, não é possível descartar que o `atrous_aspp`
apenas precise de mais tempo para convergir — o experimento não isola as duas hipóteses.

### Resultados — Eixo 2 (decoder `skip`, 2 seeds)

| gamma | Sem peso — mAP / IoU fronteira | Com peso (balanceada) — mAP / IoU fronteira |
|---|---|---|
| 0 | 0.3653 ± 0.0168 / 0.4253 ± 0.0316 | 0.2475 ± 0.0512 / 0.4281 ± 0.0438 |
| 1 | 0.3613 ± 0.0534 / 0.4142 ± 0.0516 | 0.2798 ± 0.0122 / 0.4655 ± 0.0004 |
| 2 | 0.3948 ± 0.0156 / 0.4245 ± 0.0058 | 0.2752 ± 0.0577 / 0.4421 ± 0.0473 |
| 5 | 0.2789 ± 0.0105 / 0.3421 ± 0.0072 | 0.3566 ± 0.0190 / 0.4923 ± 0.0206 |

Ver `resultados/imagens/eixo2_loss_ablation.png`. Sem peso por classe, aumentar `gamma`
**piora** a classe fronteira (o termo de foco `(1-p)^γ` da Focal Loss prioriza exemplos
difíceis — mas difícil não é sinônimo de raro, então sem compensar pelo peso, γ alto só
desestabiliza o treino: `gamma=5` sem peso é a pior configuração de toda a tabela). Com
peso por classe, aumentar `gamma` **melhora** a fronteira continuamente, culminando na
melhor configuração da tabela (`gamma=5` balanceada: IoU fronteira 0.4923). O mAP de
instância nem sempre acompanha o IoU de fronteira isoladamente — balancear a classe rara
pode custar calibração nas outras classes, o que afeta a decodificação por watershed como
um todo.

## Parte 4 — Inferência em mosaico

Constrói uma imagem grande (mosaico 2×2 de imagens de validação, `mosaic.py`), roda o
modelo da Parte 2 em tiles sobrepostos (`tiling.py`), e mostra o que acontece com núcleos
que caem exatamente na costura entre dois tiles — e como corrigir isso.

**O problema**: cada tile é decodificado por watershed de forma **independente** — os ids
de instância de um tile não têm relação nenhuma com os de outro. Um núcleo que cai bem na
costura entre dois tiles vira dois ids diferentes na costura ingênua (cada tile só
contribui com sua região central pro canvas final, seguindo a prática do slide 83).

**A correção**: pra cada par de tiles vizinhos (horizontal ou verticalmente), a faixa de
sobreposição entre eles é a única região onde os dois tiles viram exatamente o mesmo
pedaço de imagem, cada um com sua própria opinião sobre quem é quem ali. Perto da linha de
corte real (não na faixa de sobreposição inteira — ver ressalva abaixo), compara-se o IoU
de cada par (instância do tile esquerdo/de cima, instância do tile direito/de baixo); pares
com IoU alto o suficiente são fundidos via Union-Find. A fusão vertical reaproveita a
mesma função da horizontal, transpondo as máscaras — evitando duas implementações
independentes do mesmo cálculo de IoU.

```bash
python mosaic.py                # (opcional) nao tem __main__ de teste isolado; usado por run_tilling_experiment.py
python run_tilling_experiment.py # monta o mosaico, roda tiles, compara costura ingenua vs. com fusao
```

### Resultados

| Método | mAP | Erro de contagem |
|---|---|---|
| Costura ingênua (sem fusão) | 0.3348 | 13 |
| Com fusão de tiles | 0.3900 | 6 |

Ver `resultados/imagens/tiling_seam_comparison.png`: na costura vertical em col=88, um
núcleo que é um objeto só no ground truth aparece partido em duas cores na costura ingênua,
e volta a ser um objeto só depois da fusão — a evidência visual do problema e da correção.

**Ressalva importante sobre a metodologia** (descoberta durante a implementação, ver
`AI_LOG.md`): a janela usada para decidir quais pares de instâncias são candidatos a fusão
precisa ser **estreita** (poucos pixels ao redor da linha de corte exata), não a faixa de
sobreposição inteira — do contrário, núcleos inteiros que só por acaso estão na região de
sobreposição (sem nunca terem sido cortados) são fundidos incorretamente. Além disso, o
tamanho do tile e o stride escolhidos não podem coincidir com as bordas das imagens de
origem coladas no mosaico (isso faz o modelo ver uma descontinuidade falsa *dentro* de um
único tile, já que a Parte 4 monta o "mosaico grande" colando imagens do dataset que na
realidade não têm relação espacial nenhuma entre si) nem usar sobreposição excessiva
(mais de ~50% infla artificialmente a contagem de fusões, mesmo sem prejudicar o resultado
final).

## Parte 5 — Galeria de falhas

### Campo receptivo teórico (obrigatório)

`receptive_field.py` calcula o campo receptivo teórico do encoder ResNet34 (fórmula
padrão dos slides 35-38: cada camada expande o campo receptivo pelo kernel efetivo —
considerando dilatação — escalado pelo salto acumulado dos strides anteriores):

```bash
python receptive_field.py
```

| Configuração | Campo receptivo | Output stride |
|---|---|---|
| Encoder completo, sem atrous (`skip`) | 899px | 32 |
| Encoder completo, com atrous (`atrous_aspp`) | 947px | 8 |
| Encoder raso, sem atrous, mesmo output stride 8 (parado após `layer2`) | 179px | 8 |

Na mesma resolução de saída (output stride 8), o atrous aumenta o campo receptivo de
179px para 947px sem perder resolução espacial — exatamente o ganho que o slide 40
descreve. **Ressalva**: campo receptivo *teórico* cresce rápido e costuma superar o
tamanho da imagem em redes fundas (fato conhecido — o campo receptivo *efetivo*, que
realmente influencia a predição, é bem menor); reportamos o teórico porque é o que o
enunciado pede.

Comparando com a distribuição real de tamanho dos núcleos (maior lado do bounding box,
29.212 núcleos analisados): mediana de **7px**, percentil 90 de **15px**, máximo
observado de **59px**. Ou seja, **nenhum núcleo chega perto de exceder o campo
receptivo** — nem o raso. O diagnóstico-exemplo do enunciado ("objeto maior que o campo
receptivo") não se aplica a este dataset: o gargalo não é campo de visão, é resolução
espacial (mediana de 7px é menor que o *output stride* de 32px do `skip`, e quase do
tamanho do stride de 8px do `atrous_aspp`).

### As 5 piores imagens

`failure_gallery.py` roda o modelo final (Trilha A, checkpoint da Parte 2) em todas as
134 imagens de validação, calcula o mAP por imagem, e gera a figura de 4 painéis (entrada
/ ground truth / predição / mapa de probabilidade de fronteira) para as 5 piores:

```bash
python failure_gallery.py
```

| # | Núcleos reais | Previstos | Tamanho (min-max) | Diagnóstico |
|---|---|---|---|---|
| 1 (idx 19) | 68 | 7 | 1-5px | Imagem nativa **1272×603px**, redimensionada para 128×128 (~10x de encolhimento). Núcleos minúsculos e densamente empacotados; o mapa de fronteira reconhece a região do aglomerado (ver figura) mas não tem resolução para desenhar dezenas de fronteiras individuais entre vizinhos a poucos pixels de distância. |
| 2 (idx 42) | 2 | 5 | 9-10px | Imagem de entrada com contraste quase nulo (praticamente uniforme). O canal de fronteira "alucina" padrões em anel a partir de ruído de fundo, criando instâncias fantasmas em regiões sem núcleo nenhum. |
| 3 (idx 47) | 1 | 3 | 12px | Núcleo único, muito fraco, em imagem extremamente escura/ruidosa. A rede localiza aproximadamente o núcleo real mas também alucina 2 fragmentos extras a partir de ruído. |
| 4 (idx 49) | 289 | 4 | 1-5px | Mesma resolução nativa do caso 1 (1272×603px) — o caso de densidade mais extremo do dataset inteiro. A rede acerta a forma geral (um anel), mas colapsa as 289 instâncias reais em ~4 blobs: o mesmo problema de resolução do caso 1, amplificado pela densidade. |
| 5 (idx 123) | 2 | 2 | 6-7px | Dois núcleos minúsculos e adjacentes; a contagem bate por coincidência (os dois se fundem numa marca de watershed só, enquanto um artefato de ruído próximo é contado como núcleo à parte) — a contagem total dá certo, mas a correspondência espacial real falha nos dois, daí mAP=0. |

Ver as figuras em `resultados/imagens/failure_case_0{1..5}_idx*.png`.

### Correção implementada: aumentar a resolução de entrada

Os casos 1 e 4 (as duas imagens de resolução nativa 1272×603, ~10x de encolhimento) e o
caso 5 (resolução nativa 256×256, ~2x de encolhimento) apontam para o mesmo mecanismo:
resolução espacial perdida no pré-processamento (redimensionar tudo para 128×128 antes de
qualquer processamento pela rede), não limitação de campo receptivo. A correção testada:
retreinar a Trilha A com `image_size=(256, 256)` em vez de `(128, 128)`, mantendo todo o
resto igual (decoder `skip`, CE balanceada, erosão adaptativa).

```bash
python -c "from train_instance_head import train_instance_head; train_instance_head(image_size=(256,256), checkpoint_path=r'resultados/modelos/best_instance_head_model_256.pth')"
python resolution_fix_comparison.py
```

| Caso (val_idx) | Núcleos | mAP 128px | mAP 256px | Erro 128px | Erro 256px |
|---|---|---|---|---|---|
| 19 | 68 | 0.0000 | 0.0000 | 61 | 58 |
| 42 | 2 | 0.0000 | 0.0000 | 3 | 4 |
| 47 | 1 | 0.0000 | 0.0000 | 2 | 2 |
| 49 | 289 | 0.0000 | 0.0003 | 285 | 229 |
| 123 | 2 | 0.0000 | **0.3667** | 0 | 0 |

**Resultado honesto, não um "funcionou tudo"**: no caso 5 (downscale nativo de só 2x), a
correção resolveu quase completamente — confirmação direta do diagnóstico. Nos casos 1 e 4
(downscale nativo de ~10x), dobrar a resolução alvo ajuda (erro de contagem cai) mas não o
suficiente para gerar mAP relevante — em 256px ainda estamos ~5x menores que o nativo, ou
seja, a correção acertou a *direção* do diagnóstico mas não a *magnitude* necessária para
esses casos extremos (provavelmente precisariam de resolução ainda mais próxima da nativa,
ou de um processamento em tiles menores só para os aglomerados mais densos — conectando de
volta com a Parte 4). Nos casos 2 e 3 (baixo contraste/ruído), a correção não mudou nada,
resolução não é uma correção universal para todo tipo de falha.

## Parte 6 — Teste de estresse (corrupções de imagem)

Avaliação de robustez da Trilha A (modelo final de 3 classes + watershed) sob corrupções sintéticas aplicadas na imagem de entrada durante a inferência. Foram testados **3 tipos de corrupção** em **3 níveis de intensidade** (*leve*, *moderada*, *forte*):

1. **Blur (desfoque gaussiano)**: simula desfoque de lente ($\sigma \in \{1.0, 2.5, 5.0\}$).
2. **Ruído gaussiano aditivo**: simula ruído de sensor térmico/óptico ($\text{std} \in \{0.05, 0.15, 0.30\}$).
3. **Brilho/Contraste**: simula variação de iluminação do espécime ($\text{fator de contraste} \in \{0.8, 0.6, 0.35\}$, $\text{deslocamento de brilho} \in \{-0.05, -0.12, -0.20\}$).

```bash
python stress_test.py
```

Roda a avaliação em todas as 134 imagens de validação para cada configuração, salva os resultados numéricos em `resultados/stress_test_resultados.csv` e gera a figura `resultados/imagens/stress_test_corruptions.png`.

### Resultados

| Corrupção | Nível | Parâmetro | mAP @ [0.50:0.95] | Erro de contagem médio |
|---|---|---|---|---|
| Baseline | 0 | Sem corrupção | 0.2862 | 12.13 |
| Blur | 1 (leve) | $\sigma=1.0$ | 0.1758 | 15.21 |
| Blur | 2 (moderada) | $\sigma=2.5$ | 0.0907 | 23.19 |
| Blur | 3 (forte) | $\sigma=5.0$ | 0.0283 | 40.67 |
| Ruído gaussiano | 1 (leve) | $\text{std}=0.05$ | 0.0768 | 156.73 |
| Ruído gaussiano | 2 (moderada) | $\text{std}=0.15$ | 0.0297 | 192.28 |
| Ruído gaussiano | 3 (forte) | $\text{std}=0.30$ | 0.0099 | 123.30 |
| Brilho/contraste | 1 (leve) | $\text{contraste}=0.8, \text{brilho}=-0.05$ | 0.3039 | 12.68 |
| Brilho/contraste | 2 (moderada) | $\text{contraste}=0.6, \text{brilho}=-0.12$ | 0.3048 | 13.43 |
| Brilho/contraste | 3 (forte) | $\text{contraste}=0.35, \text{brilho}=-0.20$ | 0.2546 | 14.80 |

Ver a curva de degradação em `resultados/imagens/stress_test_corruptions.png`.

### Análise

- **Ruído gaussiano (vulnerabilidade crítica)**: Foi a corrupção mais destrutiva para o pipeline de watershed. O ruído aditivo no fundo gera pequenas flutuações de alta frequência que o canal de interior interpreta como dezenas de "nascentes" falsas de marcadores. Como resultado, o erro de contagem dispara para mais de 150 células a mais por imagem (super-segmentação maciça), colapsando o mAP para 0.0768 no nível leve.
- **Blur (degradação suave)**: O desfoque reduz o mAP de forma contínua conforme $\sigma$ aumenta. Como a classe fronteira entre dois núcleos vizinhos tem apenas 1 a 2 pixels de espessura, a suavização gaussiana mistura a resposta de fronteira com o interior, fundindo marcadores e fazendo o erro de contagem subir por sub-segmentação.
- **Brilho e contraste (alta robustez)**: O modelo manteve o mAP praticamente inalterado ($\sim 0.25 - 0.30$), demonstrando que a U-Net e a normalização de entrada são altamente invariantes a escurecimento ou queda de contraste uniforme.

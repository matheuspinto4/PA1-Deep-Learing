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

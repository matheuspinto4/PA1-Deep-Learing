# AI_LOG

Usamos o Claude Code (assistente de IA) ao longo de todo o PA1, em modo de par-programação:
pedíamos explicações e o código era escrito por nós mesmos (copiado das sugestões, revisado
e adaptado), exceto documentação (README, este log) e alguns arquivos de configuração, que
o assistente escreveu diretamente. Abaixo, os episódios mais relevantes — principalmente os
que envolveram debugging real, não só geração de código de primeira tentativa.

## Reorganização do projeto antes da Parte 2

Antes de implementar a Parte 2, paramos para discutir estrutura: o `evaluate_baseline.py`
da Parte 1 tinha a lógica de mAP/matching guloso embutida, e a `DiceLoss`/métricas
semânticas estavam duplicadas entre `train.py` e `train_synthetic.py`. Como a Parte 2, a
Parte 3 e a Parte 4 precisariam reusar exatamente a mesma métrica de instância (pra
comparação ser válida), refatoramos essa lógica para `metrics.py`, `losses.py` e
`postprocessing.py` antes de continuar, em vez de duplicar de novo em cada parte nova.

## Espessura de erosão do rótulo de fronteira (Trilha A, Parte 2)

Na primeira versão de `instance_labels.py`, a espessura de erosão (`border_size`) era
fixa (2px) para todo núcleo, não importa o tamanho. O primeiro treino da Trilha A deu
resultado pior que o esperado numa imagem muito densa (123 núcleos), então escrevemos um
diagnóstico (`check_border_size.py`) que mostrou que **40,1% dos núcleos do dataset
perdiam o interior por completo** com essa erosão fixa (a área média dos núcleos que
perdiam o interior era de só 13,7px², muito menor que o núcleo médio). A correção foi usar
a transformada de distância pra calcular um teto adaptativo de erosão por núcleo (nunca
erodir mais do que o "raio inscrito" da forma permite), o que reduziu a perda de interior
para 3,7% e melhorou o mAP da Trilha A de 0,26 para 0,34 no mesmo conjunto de validação.

## Generalizando CE, CE balanceada, focal e focal balanceada numa loss só (Parte 3)

Percebemos, olhando o slide 78 dos slides da aula, que a curva de γ=0 da Focal Loss é
idêntica à Cross-Entropy comum — então em vez de implementar 4 losses separadas para o
Eixo 2 de ablação, implementamos uma `FocalLoss(gamma, weight)` só, e validamos isso com um
teste de sanidade (`losses.py`, bloco `__main__`): comparamos `FocalLoss(gamma=0, weight=W)`
contra `nn.CrossEntropyLoss(weight=W)` com o mesmo tensor de entrada e conferimos que os
valores batem (`torch.allclose`), antes de confiar nela pro sweep de ablação inteiro.

## `atrous_aspp` no ResNet34: o `BasicBlock` do torchvision recusa dilatação

Para o Eixo 1 (mecanismo de recuperação de resolução), tentamos usar o argumento
`replace_stride_with_dilation` do `torchvision.models.resnet34` — só descobrimos, ao ler a
mensagem de erro, que o `BasicBlock` (usado no ResNet34, diferente do `Bottleneck` dos
ResNets maiores) recusa `dilation > 1` no construtor. A solução foi construir o ResNet34
normal e editar os atributos `stride`/`dilation`/`padding` das convoluções já construídas
diretamente (uma técnica que outras implementações públicas de "dilated ResNet" também
usam) — sem precisar trocar de encoder, o que preservou a comparação "no mesmo encoder"
pedida no enunciado.

## Conflito de nome de arquivo: `tilling.py` vs. o pacote `tiling` do PyPI

Ao criar o arquivo `tiling.py` da Parte 4, salvamos com um typo (`tilling.py`, com dois
"L"). O `import tiling` de outro script não deu `ModuleNotFoundError` — deu um
`ImportError` estranho, porque **existe um pacote de verdade no PyPI chamado `tiling`**
(`ImageTilingUtils`, sem nenhuma relação com o nosso código) instalado no ambiente Python.
Sem o `tiling.py` local pra sombrear esse pacote, o Python importou o pacote errado.
Corrigido renomeando o arquivo.

## A fusão de instâncias entre tiles (Parte 4): quatro tentativas até acertar

Esse foi o episódio de debugging mais longo do projeto. Implementamos a correção da
Parte 4 (fundir instâncias cortadas por uma costura de tile, via IoU + Union-Find) e o
primeiro teste devolveu 51 fusões para 168 núcleos reais — número absurdo, e muitas com
IoU exatamente 1,00. Depurando com a IA, identificamos e corrigimos, em sequência:

1. **Janela ampla demais para decidir candidatos**: a primeira versão comparava
   instâncias em toda a faixa de sobreposição entre tiles (até 80px de largura) — só que
   um núcleo típico tem 8-15px de diâmetro, então núcleos inteiros que nunca foram
   cortados, mas calharam de estar na região de sobreposição, eram fundidos incorretamente.
   Corrigido com duas margens separadas: uma estreita (poucos pixels) pra decidir quem é
   candidato (precisa tocar a linha de corte de verdade), outra mais larga só pra calcular
   o IoU de forma mais robusta uma vez que o candidato já foi selecionado.
2. **Mosaico com descontinuidade falsa dentro de um tile**: ao expandir de uma fileira 1×4
   pra uma grade 2×2 de verdade, o número de fusões voltou a explodir (48) e apareceu uma
   listra de artefato horizontal na visualização, presente tanto na costura ingênua quanto
   na com fusão (sinal de que não era bug de fusão). A causa: com exatamente 2 tiles
   cobrindo um eixo, o ponto médio da sobreposição cai sempre no centro exato do eixo — que,
   por coincidência, é exatamente onde duas imagens de origem (coladas para formar o
   mosaico) se encontram. Um tile cujo recorte cru inclui pedaços de duas imagens sem
   relação nenhuma confunde o modelo, que nunca viu esse tipo de descontinuidade artificial
   no treino. Corrigido usando 3 tiles por eixo com parâmetros escolhidos para que as
   costuras não coincidissem com as bordas das imagens de origem.
3. **Tamanho de tile não-múltiplo de 32**: a escolha seguinte de tile (140px) quebrou o
   decoder com skip connections (`RuntimeError` de concatenação) — o encoder reduz a
   resolução por um fator de 32 (2⁵, cinco estágios de stride 2), e um tamanho de entrada
   que não é múltiplo de 32 produz arredondamentos diferentes em cada estágio, que não
   batem na hora de concatenar com as skip connections. Corrigido voltando a um tamanho
   múltiplo de 32 (160px) que já tinha funcionado antes.
4. **Sobreposição excessiva (70%) inflando a contagem de fusões por redundância**: com
   160px de tile e stride de 48px, a sobreposição entre tiles vizinhos chegava a 70% do
   tile inteiro — cada costura acabava sendo checada várias vezes por bandas de linha/coluna
   quase idênticas, contando o mesmo par de instâncias genuinamente cortado 2-3 vezes (não
   estava errado no resultado final, já que o Union-Find deduplica, mas deixava o
   diagnóstico ilegível). Corrigido reduzindo o tile para 96px com 17% de sobreposição —
   mais parecido com a prática descrita no slide 83 (sobreposição modesta, não a maior parte
   do tile se repetindo).

O resultado final: mAP subindo de 0,3348 (costura ingênua) para 0,3900 (com fusão), erro
de contagem caindo de 13 para 6, e uma inspeção visual (`tiling_seam_comparison.png`)
confirmando um núcleo específico que aparece partido em dois ids na costura ingênua e volta
a ser um objeto só depois da correção.

## O gargalo real não é campo receptivo, é resolução de pré-processamento (Parte 5)

O enunciado dá um exemplo de diagnóstico ("o objeto tem 180px e o campo receptivo é
140px"). Calculamos o campo receptivo teórico do nosso encoder antes de assumir que esse
exemplo se aplicaria: mesmo a versão mais rasa do ResNet34 (parada na `layer2`, mesmo
output stride do `atrous_aspp`) já tem campo receptivo teórico de 179px — maior que a
imagem inteira (128×128). Comparando com a distribuição real de tamanho dos núcleos
(mediana de 7px, máximo de 59px), ficou claro que nenhum núcleo chega perto de exceder o
campo receptivo — o exemplo do enunciado não se aplica a este dataset.

Isso mudou a direção da investigação: em vez de procurar objetos "grandes demais", rodamos
o modelo final nas 134 imagens de validação e olhamos as 5 piores por mAP. Os dois casos
mais catastróficos (68 e 289 núcleos reais, quase todos previstos como um blob só)
tinham uma coisa em comum ao checar a resolução *original* das imagens antes do nosso
resize fixo para 128×128: ambas eram nativamente **1272×603px** — um encolhimento de quase
10x. Isso sugeriu que o gargalo não era a arquitetura, era o pré-processamento jogando fora
resolução antes da rede processar qualquer coisa. Testamos retreinando com
`image_size=(256,256)` e comparando as mesmas 5 imagens antes/depois: o caso com menor
encolhimento nativo (2x) teve uma melhora dramática (mAP 0 → 0,367); os dois casos de
encolhimento nativo de 10x melhoraram no erro de contagem mas não o suficiente pra gerar
mAP relevante (256px ainda é ~5x menor que o nativo); e os dois casos restantes (baixo
contraste/ruído, não resolução) não mudaram nada — confirmando que tinham uma causa raiz
diferente. Um resultado nuançado, não um "funcionou tudo", mas que confirma a direção do
diagnóstico com uma correção real e medida.

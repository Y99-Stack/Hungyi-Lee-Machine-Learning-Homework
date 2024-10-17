[toc]

# Task

Machine translation 中文翻译为英文

此任务没有什么难度，跟着课件中的hint做即可

**Data**

- Paired data
  - TED2020: TED Talk
    - Raw: 400,726 (sentences)
    - Processed: 394, 052 (sentences)
  - en, zh
- Monolingual data
  - traditional Chinese(zh)

**Workflow**

![image-20241017230417786](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017230417786.png)

1. Preprocessing  

   a. download raw data  
   b. clean and normalize  
   c. remove bad data (too long/short)  
   d. tokenization  

2. Training  

   a. initialize a model  
   b. train it with training data  

3. Testing  

   a. generate translation of test data  
   b. evaluate the performance

# Model

## Encoder

- The Encoder is a RNN or Transformer Encoder. The following description is for RNN. For every input token, Encoder will generate a output vector and a hidden states vector, and the hidden states vector is passed on to the next step. In other words, the Encoder sequentially reads in the input sequence, and outputs a single vector at each timestep, then finally outputs the final hidden states, or content vector, at the last timestep.

- Parameters:
  - *args*
      - encoder_embed_dim: the dimension of embeddings, this compresses the one-hot vector into fixed dimensions, which achieves dimension reduction
      - encoder_ffn_embed_dim is the dimension of hidden states and output vectors
      - encoder_layers is the number of layers for Encoder RNN
      - dropout determines the probability of a neuron's activation being set to 0, in order to prevent overfitting. Generally this is applied in training, and removed in testing.
  - *dictionary*: the dictionary provided by fairseq. it's used to obtain the padding index, and in turn the encoder padding mask. 
  - *embed_tokens*: an instance of token embeddings (nn.Embedding)

- Inputs: 
    - *src_tokens*: integer sequence representing english e.g. 1, 28, 29, 205, 2 
- Outputs: 
    - *outputs*: the output of RNN at each timestep, can be furthur processed by Attention
    - *final_hiddens*: the hidden states of each timestep, will be passed to decoder for decoding
    - *encoder_padding_mask*: this tells the decoder which position to ignore


* 编码器（Encoder）是一个循环神经网络（RNN）或者 Transformer 中的编码器。下面的描述是针对 RNN 的。对于每一个输入的 token，编码器会生成一个输出向量和一个隐藏状态向量，并且将隐藏状态向量传递给下一步。换句话说，编码器顺序地读入输入序列，并且在每一个时间步输出一个单独的向量，然后在最后一个时间步输出最终的隐藏状态，或者称为内容向量（content vector）。

* 参数:
    - *args*
        - encoder_embed_dim: 嵌入的维度，将 one-hot 向量压缩到固定的维度，实现降维的效果
        - encoder_ffn_embed_dim: 隐藏状态和输出向量的维度
        - encoder_layers: RNN 编码器的层数
        - dropout 确定了一个神经元的激活值被设为 0 的概率，用于防止过拟合。通常这个参数在训练时使用，在测试时移除
    - *dictionary*: fairseq 提供的字典。它用于获取填充索引，进而得到编码器的填充掩码（encoder padding mask）
    - *embed_tokens*: 一个 token embedding 的实例（nn.Embedding）

* Inputs:
    - src_tokens: 一个表示英语的整数序列，例如: 1, 28, 29, 205, 2

* Outputs:
    - outputs: RNN 在每个时间步的输出，可以由注意力机制（Attention）进一步处理
    - final_hiddens: 每个时间步的隐藏状态，会被传递给解码器（decoder）进行解码
    - encoder_padding_mask: 这个参数告诉解码器哪些位置要忽略

## Attention

- When the input sequence is long, "content vector" alone cannot accurately represent the whole sequence, attention mechanism can provide the Decoder more information.
- According to the **Decoder embeddings** of the current timestep, match the **Encoder outputs** with decoder embeddings to determine correlation, and then sum the Encoder outputs weighted by the correlation as the input to **Decoder** RNN.
- Common attention implementations use neural network / dot product as the correlation between **query** (decoder embeddings) and **key** (Encoder outputs), followed by **softmax**  to obtain a distribution, and finally **values** (Encoder outputs) is **weighted sum**-ed by said distribution.

- Parameters:
  - *input_embed_dim*: dimensionality of key, should be that of the vector in decoder to attend others
  - *source_embed_dim*: dimensionality of query, should be that of the vector to be attended to (encoder outputs)
  - *output_embed_dim*: dimensionality of value, should be that of the vector after attention, expected by the next layer

- Inputs: 
    - *inputs*: is the key, the vector to attend to others
    - *encoder_outputs*:  is the query/value, the vector to be attended to
    - *encoder_padding_mask*: this tells the decoder which position to ignore
- Outputs: 
    - *output*: the context vector after attention
    - *attention score*: the attention distribution


- 当输入序列很长时，单独的“内容向量”就不能准确地表示整个序列，注意力机制可以为解码器提供更多信息。

- 根据当前时间步的解码器embeddings，将编码器输出与解码器 embeddings 进行匹配，确定相关性，然后将编码器输出按相关性加权求和作为解码器 RNN 的输入。

- 常见的注意力实现使用神经网络/点积作为 query（解码器 embeddings）和 key（编码器输出）之间的相关性，然后用 softmax 得到一个分布，最后用该分布对 value（编码器输出）进行加权求和。

- 参数:
    - input_embed_dim: key 的维度，应该是解码器中用于 attend 其他向量的向量的维度
    - source_embed_dim: query 的维度，应该是被 attend 的向量（编码器输出）的维度
    - output_embed_dim: value 的维度，应该是 after attention 的向量的维度，符合下一层的期望,

- Inputs:
    - inputs: key, 用于 attend 其他向量
    - encoder_outputs: query/value, 被 attend 的向量
    - encoder_padding_mask: 这个告诉解码器应该忽略那些位置

- Outputs:
    - output: attention 后的上下文向量
    - attention score: attention 的分数

## Decoder

* The hidden states of **Decoder** will be initialized by the final hidden states of **Encoder** (the content vector)
* At the same time, **Decoder** will change its hidden states based on the input of the current timestep (the outputs of previous timesteps), and generates an output
* Attention improves the performance
* The seq2seq steps are implemented in decoder, so that later the Seq2Seq class can accept RNN and Transformer, without furthur modification.
- Parameters:
  - *args*
      - decoder_embed_dim: is the dimensionality of the decoder embeddings, similar to encoder_embed_dim，
      - decoder_ffn_embed_dim: is the dimensionality of the decoder RNN hidden states, similar to encoder_ffn_embed_dim
      - decoder_layers: number of layers of RNN decoder
      - share_decoder_input_output_embed: usually, the projection matrix of the decoder will share weights with the decoder input embeddings
  - *dictionary*: the dictionary provided by fairseq
  - *embed_tokens*: an instance of token embeddings (nn.Embedding)
- Inputs: 
    - *prev_output_tokens*: integer sequence representing the right-shifted target e.g. 1, 28, 29, 205, 2 
    - *encoder_out*: encoder's output.
    - *incremental_state*: in order to speed up decoding during test time, we will save the hidden state of each timestep. see forward() for details.
- Outputs: 
    - *outputs*: the logits (before softmax) output of decoder for each timesteps
    - *extra*: unsused


- 解码器的隐藏状态将由编码器的最终隐藏状态（the content vector）初始化
- 同时，解码器会根据当前时间步的输入（前一时间步的输出）改变其隐藏状态，并生成一个输出
- 注意力机制可以提高性能
- seq2seq 的步骤是在解码器中实现的，这样以后 Seq2Seq 类可以接受 RNN 和 Transformer，而不需要进一步修改。

- 参数:
    - args
        - decoder_embed_dim: 解码器嵌入的维度，类似于 encoder_embed_dim
        - decoder_ffn_embed_dim: 解码器 RNN 隐藏状态的维度，类似于 encoder_ffn_embed_dim
        - decoder_layers: RNN 解码器的层数
        - share_decoder_input_output_embed: 通常，解码器的投影矩阵会与解码器输入 embeddings 共享权重
    - dictionary: fairseq 提供的字典
    - embed_tokens: 一个 token embedding 的实例（nn.Embedding）
- 输入:
    - prev_output_tokens: 表示右移目标的整数序列，例如: 1, 28, 29, 205, 2
    - encoder_out: 编码器的输出
    - incremental_state: 为了加速测试时的解码，我们会保存每个时间步的隐藏状态。详见forward()。
- 输出:
    - outputs: 解码器在每个时间步的输出的对数（softmax之前）
    - extra: 未使用

# Evaluation

## BLEU（双语评估替换）

BLEU（Bilingual Evaluation Understudy）是一种用于评估机器生成的翻译质量的指标，通常与一个或多个参考翻译进行比较。它由Kishore Papineni等人在2002年的论文《BLEU: a Method for Automatic Evaluation of Machine Translation》中提出。

关键概念

1. **N-gram 匹配**
   - BLEU通过计算机器翻译与参考翻译之间的N-gram匹配来评估翻译质量。N-gram是指连续的N个词的序列。
   - 常见的N-gram包括：
     - 1-gram（unigram）：单个词的匹配。"Thank", "you", "so", "much"
     - 2-gram（bigram）：两个连续词的匹配。"Thank you", "you so", "so much"
     - 3-gram（trigram）：三个连续词的匹配。"Thank you so", "you so much"
     - 4-gram（quadgram）：四个连续词的匹配。"Thank you so much"
   
2. **精度（Precision）**
   - 精度是指机器翻译中与参考翻译匹配的N-gram的比例。
   - 公式：
     $
     \text{Precision} = \frac{\text{机器翻译中匹配的N-gram数量}}{\text{机器翻译中所有N-gram数量}}
     $
   
 3. **修正精度（Modified Precision）**
   - 为了避免机器翻译中出现重复匹配的问题，BLEU引入了修正精度。
   
   - 修正精度考虑了每个N-gram在参考翻译中出现的最大次数。
     假设我们有一个机器翻译（MT）和一个参考翻译（Ref），我们想要计算N-gram的修正精度。
   
     设 $ \text{Count}_{\text{MT}}(n\text{-gram}) $ 表示某个N-gram在机器翻译中出现的次数。设 $ \text{Count}_{\text{Ref}}(n\text{-gram}) $ 表示某个N-gram在参考翻译中出现的最大次数。修正后的N-gram计数 $ \text{Count}_{\text{clip}}(n\text{-gram}) $ 取机器翻译中的N-gram计数和参考翻译中的最大N-gram计数的较小值。公式：
     $
     \text{Count}_{\text{clip}}(n\text{-gram}) = \min(\text{Count}_{\text{MT}}(n\text{-gram}), \text{Count}_{\text{Ref}}(n\text{-gram}))
     $
   
     修正精度 $ p_n $ 是所有N-gram的修正计数之和除以机器翻译中所有N-gram的计数之和。公式：
     $
     p_n = \frac{\sum_{\text{n-gram} \in \text{MT}} \text{Count}_{\text{clip}}(n\text{-gram})}{\sum_{\text{n-gram} \in \text{MT}} \text{Count}_{\text{MT}}(n\text{-gram})}
     $
   
 4. **短句惩罚（Brevity Penalty）**
   - 由于较短的翻译更容易获得高精度，BLEU引入了短句惩罚来平衡这一问题。
   - 短句惩罚的公式：
     $
     \text{BP} = \begin{cases} 
     1 & \text{如果机器翻译长度 > 参考翻译长度} \\
     e^{(1 - \frac{\text{参考翻译长度}}{\text{机器翻译长度}})} & \text{如果机器翻译长度 ≤ 参考翻译长度}
     \end{cases}
     $
   
 5. **BLEU分数**
   - BLEU分数综合了1-gram到4-gram的修正精度和短句惩罚。
   - 公式：
     $$     \text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)     $$
     其中，$ p_n $ 是N-gram的修正精度，$ w_n $ 是权重，通常取 $ w_n = \frac{1}{N} $。



# Baseline

## Simple 

Just run sample code, 在对应的ipynb文件中我已添加各个结构的详细介绍

## Medium

**Add learning rate scheduler and train longer**

- learning rate scheduler

$$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$

```python
import math

def get_rate(d_model, step_num, warmup_step):
    # TODO: Change lr from constant to the equation shown above 
    # medium
    lr = 1.0 / math.sqrt(d_model) * min(1.0 / math.sqrt(step_num), step_num / (warmup_step * math.sqrt(warmup_step)))
    return lr
```

![image-20241017133331799](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017133331799.png)

- change maximum epoch for training (15->30)

```python
config = Namespace(
    
    ...
    
    # maximum epochs for training (Medium)
    max_epoch=30,
    start_epoch=1,
    
    ...
    
)
```

## Strong

Switch to Transformer and tuning hyperparameter

- 将模型转变为Transformer(Model Initialization)

```python
# # HINT: transformer architecture
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)

def build_model(args, task):
    """ build a model instance based on hyperparameters """
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())

    # encoder decoder
    # HINT: TODO: switch to TransformerEncoder & TransformerDecoder
    # encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    # strong
    encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
    
    ...
    
    
```

- 修改模型超参数(Architecture Related Configuration)

  参考[Attention is all your Need](https://arxiv.org/abs/1706.03762)表3中*transformer-base*的超参数

  为避免训练时间过长，可以修改`max_epoch `设置得更小一些，比如10,15也可以达到很好的效果

  ![image-20241017132321480](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017132321480.png)

```python
# strong
arch_args = Namespace(
    encoder_embed_dim=512,
    encoder_ffn_embed_dim=2048,
    encoder_layers=6,  
    decoder_embed_dim=512,
    decoder_ffn_embed_dim=2048,
    decoder_layers=6,  
    share_decoder_input_output_embed=True,
    dropout=0.3,
)
```

==注意：==也可以直接用课件中hint，只是将encoder_layers和decoder_layers改为4，只是得出来的效果没有那么好，此时模型的参数数量会和之前的 RNN 差不多，在 max_epoch =30 的情况下，Bleu 可以达到 23.59，如下：

```python
arch_args = Namespace(
 encoder_embed_dim=256,
 encoder_ffn_embed_dim=512,
 encoder_layers=1, # recommend to increase -> 4
 decoder_embed_dim=256,
 decoder_ffn_embed_dim=1024,
 decoder_layers=1, # recommend to increase -> 4
 share_decoder_input_output_embed=True,
 dropout=0.3,
)

```

config里面savedir文件夹位置需要改变

```python
# config里面savedir文件夹位置需要改变，否则无法loadcheckpoint的model
config = Namespace(    
	datadir = "./DATA/databin/ted2020",    
    #savedir = "./checkpoints/rnn",    
    savedir = "./checkpoints/transformer",
```

## Boss

补全**Back-translation**部分所有的TODO

**思路：**

1. 将目标语言和源语言对调，训练从中文到英文得反向模型(backward translation model)
   - `source_lang="zh"`
   - `target_lang = "en"`
2. 用反向模型(backward translation model)对单语数据(monolingual data)进行翻译，得到综合数据(synthetic data)
   - Complete TODOs in the sample code
   - All the TODOs can be completed by using commands from earlier cells
3. 利用新的数据(synthetic data)训练更strong的正向翻译模型(`max_epoch=30`就可以过boss baseline)



完整的运行流程是：

1. 将`实验配置中 ` / `Configuration for experiments` 的 **BACK_TRANSLATION** 设置为 **True** 运行 训练一个 back-translation 模型，并处理好对应的语料。
2. 将`实验配置` / `Configuration for experiments` 中的 **BACK_TRANSLATION** 设置为 **False** 运行 结合 ted2020 和 mono (back-translation) 的语料进行训练。

```python
config = Namespace(
    datadir = "./DATA/data-bin/ted2020_with_mono", #boss
    savedir = "./checkpoints/transformer-bt",

    ...

BACK_TRANSLATION = False # Boss
if BACK_TRANSLATION:
    config.datadir = "./DATA/data-bin/ted2020"
    config.savedir = "./checkpoints/transformer-back"
    config.source_lang, config.target_lang= tgt_lang, src_lang
```

在**Generate Prediction**中

```python
if not BACK_TRANSLATION: # boss
    raise
```



### TODO: clean corpus

1. remove sentences that are too long or too short 删除太长或者太短的句子
2. unify punctuation 统一标点符号

hint: you can use `clean_s() `defined above to do this

```python
def clean_mono_corpus(prefix, l, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.zh').exists():
        print(f'{prefix}.clean.zh exists. skipping clean.')
        return
    with open(f'{prefix}', 'r') as l_in_f:
        with open(f'{prefix}.clean.zh', 'w') as l_out_f:
            for s in l_in_f:
                s = s.strip()
                s = clean_s(s, l)
                s_len = len_s(s, l)
                if min_len > 0: # remove short sentence
                    if s_len < min_len:
                        continue
                if max_len > 0: # remove long sentence
                    if s_len > max_len:
                        continue
                print(s, file=l_out_f)
                

mono_data_prefix = f'{mono_prefix}/ted_zh_corpus.deduped'
clean_mono_corpus(mono_data_prefix, 'zh')
```

**Generate pseudo translation**生成伪翻译

```python
import os

zh_path = mono_prefix / 'ted_zh_corpus.deduped.clean.zh'
en_path = mono_prefix / 'ted_zh_corpus.deduped.clean.en'

if en_path.exists():
    print(f"{en_path} exists. skipping the generation of psuedo translation.")

with open(zh_path, 'r') as zh_f:
    with open(en_path, 'w') as en_f:
        for line in zh_f:
            line = line.strip()
            if line:
                # Replace with '_.' to get a pseudo translation
                pseudo = '_.'
                # Write the pseudo translation string to the target file and add a new line
                print(pseudo, file=en_f)
            # If the line is empty, just add a new line
            else:
                print(file=en_f)
```

### TODO: Subword Units

Use the spm model of the backward model to tokenize the data into subword units 用反向模型的spm模型将数据分为子词单位

hint: spm model is located at `DATA/raw-data/\[dataset\]/spm\[vocab_num\].model`

```python
def spm_encode(prefix, vocab_size, mono_prefix):
    spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))

    in_path = mono_prefix / 'ted_zh_corpus.deduped.clean'

    for lang in [src_lang, tgt_lang]:
        out_path = mono_prefix / f'mono.tok.{lang}'
        # if out_path.exists():
        #     print(f"{out_path} exists. skipping spm_encode.")
        # else:
        with open(out_path, 'w') as out_f:
            with open(f'{in_path}.{lang}', 'r') as in_f:
                for line in in_f:
                    line = line.strip()
                    tok = spm_model.encode(line, out_type=str)
                    print(' '.join(tok), file=out_f)

spm_encode(prefix, vocab_size, mono_prefix)
```

### TODO: Generate synthetic data with backward model

Add binarized monolingual data to the original data directory, and name it with "split_name" 将二进制化的单语言数据添加到原始数据目录中，并将其命名为“split_name”

ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]

then you can use `'generate_prediction(model, task, split="split_name")'` to generate translation prediction

```python
# hint: do prediction on split='mono' to create prediction_file
task.load_dataset(split="mono", epoch=1)
generate_prediction(model, task, split='mono', outfile='./prediction.txt' )
```

### TODO: Create new dataset

1. Combine the prediction data with monolingual data
2. Use the original spm model to tokenize data into Subword Units
3. Binarize data with fairseq

```python
# Combine prediction_file (.en) and mono.zh (.zh) into a new dataset.
#
# hint: tokenize prediction_file with the spm model
!cp ./prediction.txt {mono_prefix}/'ted_zh_corpus.deduped.clean.en'
spm_encode(prefix, vocab_size, mono_prefix)
# output: ./DATA/rawdata/mono/mono.tok.en & mono.tok.zh
#
# hint: use fairseq to binarize these two files again
binpath = Path('./DATA/data-bin/synthetic')
src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file = src_dict_file
monopref = './DATA/rawdata/mono/mono.tok' # or whatever path after applying subword tokenization, w/o the suffix (.zh/.en)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    !python -m fairseq_cli.preprocess\
        --source-lang 'zh'\
        --target-lang 'en'\
        --trainpref {monopref}\
        --destdir {binpath}\
        --srcdict {src_dict_file}\
        --tgtdict {tgt_dict_file}\
        --workers 2
```

# Gradescope

## Problem 1: Visualize Positional Embedding

![image-20241017232420037](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017232420037.png)

放在Confirm model weights used to generate submission模块后

```python
def get_cosine_similarity_matrix(x):
    x = x / x.norm(dim=1, keepdim=True)
    sim = torch.mm(x, x.t())
    return sim


# Get the positional embeddings from the decoder of the model
pos_emb = model.decoder.embed_positions.weights.cpu().detach()
sim = get_cosine_similarity_matrix(pos_emb)
# sim = F.cosine_similarity(pos_emb.unsqueeze(1), pos_emb.unsqueeze(0), dim=-1) # same

# Plot the heatmap of the cosine similarity matrix of the positional embeddings
plt.imshow(sim, cmap="hot", vmin=0, vmax=1)
plt.colorbar()

plt.show()
```

![image-20241017232353311](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017232353311.png)

## Problem 2: Gradient Explosion

**Clipping Gradient Norm**

![image-20241017232600342](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017232600342.png)在`config = Namespace()`中`use_wandb=True`或如下图：

![image-20241017232757299](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW05-2023\readme.assets\image-20241017232757299.png)

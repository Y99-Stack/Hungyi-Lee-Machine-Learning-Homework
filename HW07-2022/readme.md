[toc]

# Link



# Task

HW7的任务是通过BERT完成Question Answering。

**数据预处理流程梳理**

数据解压后包含3个json文件：hw7_train.json, hw7_dev.json, hw7_test.json。

DRCD: 台達閱讀理解資料集 Delta Reading Comprehension Dataset 

ODSQA: Open-Domain Spoken Question Answering Dataset 

- train: DRCD + DRCD-TTS 
  - 10524 paragraphs, 31690 questions 
- dev: DRCD + DRCD-TTS 
  - 1490 paragraphs, 4131 questions 
- test: DRCD + ODSQA 
  - 1586 paragraphs, 4957 questions

{train/dev/test}_questions:
- List of dicts with the following keys:
 - id (int)
 - paragraph_id (int)
 - question_text (string)
 - answer_text (string)
 - answer_start (int)
 - answer_end (int)

{train/dev/test}_paragraphs:
- List of strings
- paragraph_ids in questions correspond to indexs in paragraphs
- A paragraph may be used by several questions

读取这三个文件，每个文件返回相应的question数据和paragraph数据，都是文本数据，不能作为模型的输入。

利用**Tokenization**将question和paragraph文本数据先按token为单位分开，再转换为`tokens_to_ids`数字数据。Dataset选取paragraph中固定长度的片段（固定长度为150），片段需包含answer部分，然后使用`Tokenization` 以CLS + question + SEP + document+ CLS + padding(不足的补0)的形式作为训练输入。

Total sequence length = question length + paragraph length + 3 (special tokens) 
Maximum input sequence length of BERT is restricted to 512

![image-20241016191905601](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW07-2022\readme.assets\image-20241016191905601.png)

![image-20241016191943864](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW07-2022\readme.assets\image-20241016191943864.png)

**training**

![image-20241016192935182](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW07-2022\readme.assets\image-20241016192935182.png)

**testing**

对于每个窗口，模型预测一个开始分数和一个结束分数，取最大值作为答案

![image-20241016192954602](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW07-2022\readme.assets\image-20241016192954602.png)

# Baseline

## Medium

应用**linear learning rate decay+change `doc_stride`**

这里linear learning rate decay选用了两种方法

- 手动调整学习率 

  假设初始学习率为 $η_0$，总的步骤数为$T$，那么在第 $t$步时的学习率 $η_t$ 可以表示为：

  $η_t=η_0−\frac{η_0}{T}×t$

  其中：

  - $η_0$ 是初始学习率。
  - $T$是总的步骤数（total_step）。
  - $t$ 是当前的步骤数（从 0 开始计数）。

  ​	`optimizer.param_groups[0]["lr"] -= learning_rate / total_step`是$η_t=η_t−\frac{η_0}{T}η_t$，

  - `optimizer.param_groups[0]["lr"]` 对应 $η_t$。
  - `learning_rate` 对应 $η_0$。
  - `total_step` 对应 $T$。
  - `i` 对应 $t$。

  ```python
  # Medium--Learning rate dacay
  # Method 1: adjust learning rate manually
  total_step = 1000
  for i in range(total_step):
      optimizer.param_groups[0]["lr"] -= learning_rate / total_step
  ```

  

- 通过scheduler自动调整学习率

  - (recommend) [transformer](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup)
  - [torch.optim](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

  ```python
  # Method 2: Adjust learning rate automatically by scheduler
  
  # (Recommend) https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
  from transformers import get_linear_schedule_with_warmup
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
  
  # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  # 这里如果要用pytorch的ExponentialLR，一定要导入optim模块，并且前面的AdamW是从transformers中import的这里要重新import
  import torch.optim as optim
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
  ```

change `doc_stride`在QA_Dataset的时候修改段落滑动窗口的步长

```python
##### TODO: Change value of doc_stride #####
# 段落滑动窗口的步长
self.doc_stride = 30  # Medium
```



## Strong

**应用➢ Improve preprocessing ➢ Try other pretrained models**

- 尝试其他预训练[模型](https://huggingface.co/models)

比如bert-base-multilingual-case，因为它可以避免英文无法tokenization输出[UNK]，但是计算量大

```python 
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-macbert-large").to(device)
tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-macbert-large")
```

- preprocessing ，在QA_Dataset中修改截取答案的窗口

  1. 随机窗口选择 Random Window Selection      
  随机选择窗口的起始位置
      - 随机范围的下界
        `start_min = max(0, answer_end_token - self.max_paragraph_len + 1)` 答案结束位置向前移动 `self.max_paragraph_len - 1 `个标记后的位置和 0 较大的那个
      - 随机范围的上界
        `start_max = min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len)`
          - `len(tokenized_paragraph) - self.max_paragraph_len`：计算段落长度减去窗口长度后的位置，确保窗口不会超出段落末尾。
          - `min(answer_start_token, ...)`：确保上界不超过答案开始位置，避免答案被截断。
      - 随机选择
        `paragraph_start = random.randint(start_min, start_max)`在计算出的下界和上界之间随机选择一个整数作为窗口的起始位置。
      - 计算窗口结束位置
        `paragraph_end = paragraph_start + self.max_paragraph_len`确保窗口长度为 `self.max_paragraph_len`。

  2. 滑动窗口大小 Dynamic window size

  ```python
          ##### TODO: Preprocessing Strong #####
          # Hint: How to prevent model from learning something it should not learn
  
          if self.split == "train":
              # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph
              answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
              answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
  
              # A single window is obtained by slicing the portion of paragraph containing the answer
              # 在training中paragraph的截取依据的是answer的position id
              """
              mid = (answer_start_token + answer_end_token) // 2
              paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
              paragraph_end = paragraph_start + self.max_paragraph_len"""
              # Strong
              # Method 1: Random window selection
              start_min = max(0, answer_end_token - self.max_paragraph_len + 1)  # 计算答案结束位置向前移动 self.max_paragraph_len - 1 个标记后的位置
              start_max = min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len)
              start_max = max(start_min, start_max)
              paragraph_start = random.randint(start_min, start_max + 1)
              paragraph_end = paragraph_start + self.max_paragraph_len
              
              """
              # Method 2: Dynamic window size 
              # 这个会造成窗口的大小大于max_paragraph_len，那么会造成输入序列的长度不一致，后面padding也要改，这里暂不采用
              answer_length = answer_end_token - answer_start_token
              dynamic_window_size = max(self.max_paragraph_len, answer_length + 20)  # 添加一些额外的空间
              paragraph_start = max(0, min(answer_start_token - dynamic_window_size // 2, len(tokenized_paragraph) - dynamic_window_size))
              paragraph_end = paragraph_start + dynamic_window_size
              """
  
  ```


## Boss

**➢ Improve postprocessing ➢ Further improve the above hints**

**doc_stride + max_length+ learning rate scheduler + preprocessing+ postprocessing + new model + no validation**。

与strong baseline相比，最大的改变有两个，一是换pretrain model，在hugging face中搜索chinese + QA的模型，根据model card描述选择最好的模型，使用后大概提升2.5%的精度，二是更近一步的postprocessing，查看提交文件可看到很多answer包含**CLS, SEP, UNK**等字符，CLS和SEP的出现表示预测位置有误，UNK的出现说明有某些字符无法正常编码解码（例如一些生僻字），错误字符的问题均可在evaluate函数中改进，这个步骤提升了大概1%的精度。其他的修改主要是针对overfitting问题，包括减少了learning rate，提升dataset里面的paragraph max length, 将validation集合和train集合进行合并等。另外可使用的办法有ensemble，大概能提升0.5%的精度，改变random seed，也有提升的可能性。

```python
if start_index > end_index or start_index < paragraph_start or end_index > paragraph_end:
    continue
    
if '[UNK]' in answer:
    print('发现 [UNK]，这表明有文字无法编码, 使用原始文本')
    #print("Paragraph:", paragraph)
    #print("Paragraph:", paragraph_tokenized.tokens)
    print('--直接解码预测:', answer)
    #找到原始文本中对应的位置
    raw_start =  paragraph_tokenized.token_to_chars(origin_start)[0]
    raw_end = paragraph_tokenized.token_to_chars(origin_end)[1]
    answer = paragraph[raw_start:raw_end]
    print('--原始文本预测:',answer)
```


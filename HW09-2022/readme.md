# Task

## CNN explanation

11种食物图片分类，与HW3使用同一个dataset

- Bread, Diary product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup,  and Vegetables/Fruit
- ![image-20240927225952491](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927225952491.png)

训练一个CNN model用于classification，并做一些explanations

### Lime package 

[Lime](https://github.com/marcotcr/lime)

![image-20240927230008871](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927230008871.png)

### Saliency map 



What is Saliency map ?

Saliency: 顯著性

The heatmaps that highlight pixels of the input image that contribute the most in the classification task.

Ref: https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4

We put an image into the model, forward then calculate the loss referring to the label. Therefore, the loss is related to:

- image
- model parameters
- label

Generally speaking, we change model parameters to fit "image" and "label". When backward, we calculate the partial differential value of loss to model parameters. 一般来说，我们改变模型参数来拟合“图像”和“标签”。当反向时，我们计算损失对模型参数的偏微分值。

Now, we have another look. When we change the image's pixel value, the partial differential value of loss to image shows the change in the loss. We can say that it means the importance of the pixel. We can visualize it to demonstrate which part of the image contribute the most to the model's judgment. 现在，我们再看一遍。当我们改变图像的像素值时，损耗对图像的偏微分值表示损耗的变化。我们可以说这意味着像素的重要性。我们可以将其可视化，以演示图像的哪一部分对模型的判断贡献最大。

![image-20240927230714808](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927230714808.png)

### Smooth Grad 

Smooth grad 的方法是，在圖片中隨機地加入 noise，然後得到不同的 heatmap，把這些 heatmap 平均起來就得到一個比較能抵抗 noisy gradient 的結果。 

The method of Smooth grad is to randomly add noise to the image and  get different heatmaps. The average of the heatmaps would be more robust to noisy gradient. 

ref: [https://arxiv.org/pdf/1706.03825.pdf](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1706.03825.pdf)

![image-20240927230753575](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927230753575.png)

### Filter Visualization 

https://reurl.cc/mGZNbA

![image-20240927230821781](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927230821781.png)

### Integrated Gradients

https://arxiv.org/pdf/1703.01365.pdf

![image-20240927230835670](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927230835670.png)

## BERT Explanation

- Attention Visualization
- Embedding Visualization
- Embedding analysis

##Attention Visualization

https://exbert.net/exBERT.html

##Embedding Visualization

Embedding 二维化

![image-20240927231050876](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927231050876.png)

##Embedding analysis

用Euclidean distance 和 Cosine similarity 两种方法比较output embedding

下图是"果"

![image-20240927231118184](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW09-2022\readme.assets\image-20240927231118184.png)

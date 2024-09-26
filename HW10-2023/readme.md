[toc]

# Task

**黑箱攻击(Blackbox Attack)**

采用CIFAR-10数据集经助教筛选，黑箱攻击的核心是：如果你有目标网络的训练数据，你可以训练一个proxy network，用这个proxy network生成attacked objects。

# Baseline

- Simple baseline (acc <= 0.70)
  - Hints: FGSM
  - Expected running time: 1.5 mins on T4
- Medium baseline (acc <= 0.50)
  - Hints: Ensemble Attack + ensemble random few model + IFGSM
  - Expected running time: 5 mins on T4
- Strong baseline (acc <= 0.25)
  - Hints: Ensemble Attack + ensemble many models + MIFGSM
  - Expected running time: 5.5 mins on T4
- Boss baseline (acc <= 0.10)
  - Hints: Ensemble Attack + ensemble many models + DIM-MIFGSM
  - Expected running time: 8 mins on T4

主要就是以下这几个方法，具体的理论写在ipynb中了

**Implement non-targeted adversarial attack method** 

a. FGSM  
b. I-FGSM 
c. MI-FGSM 

**Increase attack transferability by Diverse input (DIM)** 

**Attack more than one proxy model - Ensemble attack**

## FGSM (Fast Gradient Sign Method (FGSM)  

$$
x^{adv}的目标是J(x^{real},y)最大\\

x^{adv} = x^{real} + ϵ ⋅ sign(\frac{\partial{J(x^{real},y)}}{\partial{x}0})
$$

## I-FGSM(Iterative Fast Gradient Sign Method)  

$$
\begin{aligned}&\boldsymbol{x}_0^{adv}=\boldsymbol{x}^{real}\\&\mathrm{for~t=1~to~num\_iter:}\\&\boxed{\boldsymbol{x}_{t+1}^{adv}=\boldsymbol{x}*_t^{adv}+\alpha\cdot\mathrm{sign}(\nabla_*{\boldsymbol{x}}J(\boldsymbol{x}_t^{adv},y))}\\&\mathrm{clip~}\boldsymbol{x}_t^{adv}\end{aligned}
$$

其中，$\alpha$是step size   

## MI-FGSM(Momentum Iterative Fast Gradient Sign Method)  

使用momentum来稳定更新方向，并摆脱糟糕的局部最大值   
$$
\begin{aligned}
&\text{for t = 1 to num iter:} \\
&\boldsymbol{g}_{t+1}=\mu\cdot\boldsymbol{g}_t+\frac{\nabla_\boldsymbol{x}J(\boldsymbol{x}_t^{adv},y)}{\|\nabla_\boldsymbol{x}J(\boldsymbol{x}_t^{adv},y)\|_1},\quad\text{decay factor }\mu \\
&\boxed{\boldsymbol{x}_{t+1}^{adv}=\boldsymbol{x}_t^{adv}+\alpha\cdot\mathrm{sign}(\boldsymbol{g}_{t+1}),} \\
&\operatorname{clip}\boldsymbol{x}_t^{adv}
\end{aligned}
$$
其中$g$是momentum   

## M-DI2-FGSM(Diverse Input Momentum Iterative Fast Gradient Sign Method)

$$T(X_n^{adv};p)=\begin{cases}T(X_n^{adv})&\text{ with probability }p\\X*_n^{adv}&\text{ with probability }1-p\end{cases}\\\text{e.g. DIM + MI-FGSM}\\g_*{n+1}=\mu\cdot g_n+\frac{\nabla_XL(T(X_n^{adv};p),y^{\mathrm{true}};\theta)}{||\nabla_XL(T(X_n^{adv};p),y^{\mathrm{true}};\theta)||_1}$$   

这里的L可以用CrossEntropyLoss求解

# Report

## fgsm attack

```python
# original image
path = f'dog/dog2.png'
im = Image.open(f'./data/{path}')
logit = model(transform(im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'benign: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(np.array(im))
plt.tight_layout()
plt.show()

# adversarial image
adv_im = Image.open(f'./fgsm/{path}')
logit = model(transform(adv_im).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')
plt.imshow(np.array(adv_im))
plt.tight_layout()
plt.show()
```

![image-20240926234008205](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW10-2023\readme.assets\image-20240926234008205.png)

## Jepg Compression

```python
import imgaug.augmenters as iaa

# pre-process image
x = transforms.ToTensor()(adv_im)*255
x = x.permute(1, 2, 0).numpy()
x = x.astype(np.uint8)

# TODO: use "imgaug" package to perform JPEG compression (compression rate = 70)
compressed_x = iaa.JpegCompression(compression=70)(image=x)

logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
predict = logit.argmax(-1).item()
prob = logit.softmax(-1)[predict].item()
plt.title(f'JPEG adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
plt.axis('off')


plt.imshow(compressed_x)
plt.tight_layout()
plt.show()
```

![image-20240926233953049](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW10-2023\readme.assets\image-20240926233953049.png)
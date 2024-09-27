[toc]

# Task

**异常检测Anomaly Detection**

![image-20240927194939917](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW08-2022\readme.assets\image-20240927194939917.png)

将data经过Encoder，在经过Decoder，根据输入和输出的差距来判断异常图像。training data是100000张人脸照片，testing data有大约10000张跟training data相同分布的人脸照片(label 0)，还有10000张不同分布的照片(anomaly, label 1)，每张照片都是(64,64,3)，`.npy file`

以训练集的前三张照片为例，auto-encoder的输入和输出如下：

![image-20240927201008427](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW08-2022\readme.assets\image-20240927201008427.png)

# Baseline

Auto-encoder model一共有五种模型

- fcn: fully-connected network
- cnn: convolutional network
- VAE
- Resnet
- Multi-encoder autoencoder
  - encoder(fcn+fcn+fcn)+decoder(fcn)
  - encoder(cnn+cnn+cnn)+decoder(cnn)
  - encoder(fcn+fcn+conv)+decoder(fcn)

通过FCN+调节超参数的方式可以轻易的达到strong，Resnet也是但是Multi-encoder的方式表现并不好，也许是我处理方式有问题，具体代码可以参考GitHub中的文件



# Report

## Question2

Train a fully connected autoencoder and adjust at least two different  element of the latent representation. Show your model architecture, plot out the original image, the reconstructed images for each adjustment  and describe the differences.

```python
import matplotlib.pyplot as plt
# sample = train_dataset[random.randint(0,100000)]
sample = train_dataset[0]
print("sample shape:{}".format(sample.size()))
sample = sample.reshape(1,3,64,64)

model.eval()
with torch.no_grad():
    img = sample.cuda()
            
    # 只调整fcn中的latent representation的其中两维，其他模型都是正常输出
    if model_type in ['res']:
        output = model(img)
        output = decoder(output)
        print("res output shape:{}".format(output.size()))
        output = output[0] # 第一个重建图像，当然只有一个图像
        
    if model_type in ['fcn']:
        img = img.reshape(img.shape[0], -1)
        x = model.encoder(img)
        x[0][2] = x[0][2]*3
        output = model.decoder(x)
        print("fcn output shape:{}".format(output.size()))
        output = output.reshape(3,64,64)
        
    if model_type in ['vae']:
        output = model(img)
        print("vae output shape:{}".format(output.size()))
        output = output[0][0] # output[0]是重建后的图像，output[0][0]重建后的第一个图像
        
    if model_type in ['cnn']:
        output = model(img)[0]
        
    print("output shape:{}".format(output.size()))
       
sample = sample.reshape(3,64,64)   

# 创建画布
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

# plt sample image
axes[0].imshow(transforms.ToPILImage()((sample+1)/2)) #imshow的输入(H,W,C)
axes[0].axis('off')
axes[0].annotate('sample input', xy=(0.5, -0.15), xycoords='axes fraction',ha='center', va='center')
# plt output image
axes[1].imshow(transforms.ToPILImage()((output+1)/2))
axes[1].axis('off')
axes[1].annotate('sample output', xy=(0.5, -0.15), xycoords='axes fraction',ha='center', va='center')

plt.show()
```

![image-20240927200012819](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW08-2022\readme.assets\image-20240927200012819.png)
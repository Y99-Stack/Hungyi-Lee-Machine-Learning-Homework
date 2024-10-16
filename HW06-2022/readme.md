[toc]

# Task

任务的目的是**生成二次元的头像(Anime face generation)**

- **Input**: random number
- **Output**: Anime face 
- **Implementation requirement**: Diffusion Model 
- **Target**: generate 1000 anime face images

## Evaluation metrics评价标准

### FID (Frechet Inception Distance) score

1. 使用另一个模型来创建真实和虚假图像的特征

2. 计算两个特征分布之间的Frechet距离

FID（Frechet Inception Distance）是一种用于评估生成模型（如生成对抗网络GANs）生成图像质量的指标。它通过比较生成图像与真实图像在特征空间中的分布差异来衡量生成图像的质量。FID越低，表示生成图像与真实图像的分布越接近，生成图像的质量越高。

FID的计算步骤如下：

- 提取特征：使用预训练的Inception网络（或其他合适的网络）从生成图像和真实图像中提取特征。通常，这些特征是从网络的某一层（如最后一层全连接层）输出的。

- 计算均值和协方差：对于生成图像和真实图像的特征，分别计算它们的均值（mean）和协方差（covariance）。

- 使用Frechet距离（也称为Wasserstein-2距离）来度量两个多元正态分布（即生成图像特征分布和真实图像特征分布）之间的距离。具体公式：$$\mathrm{FID}=\|\mu_r-\mu_g\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$$
其中，$\mu_r$和$\Sigma_r$分别是真实图像特征的均值和协方差，$\mu_g$和$\Sigma_g$分别是生成图像特征的均值和协方
差，Tr 表示矩阵的迹(trace)。

![image-20241016231006183](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW06-2022\readme.assets\image-20241016231006183.png)

```python
def calculate_fid(real_images_path, generated_images_path):
    """
    Calculate FID score between real and generated images.
    
    :param real_images_path: Path to the directory containing real images.
    :param generated_images_path: Path to the directory containing generated images.
    :return: FID score
    """
    fid = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], batch_size=50, device='cuda', dims=2048)
    return fid
```



### AFD (Anime face detection) rate

检测提交的图片中有多少张面孔，越高越好

AFD（Anime Face Detection）率是指在动漫图像或视频中检测和识别动漫角色面部特征的成功率。这个指标通常用于评估动漫面部检测算法或模型的性能。AFD率可以通过以下几个方面来衡量：

- 准确率（Accuracy）：正确检测到的动漫面部数量与总面部数量的比例。
- 召回率（Recall）：正确检测到的动漫面部数量与实际存在的面部数量的比例。
- 精确率（Precision）：正确检测到的动漫面部数量与检测到的总面部数量的比例。

AFD率的计算通常涉及以下步骤：

- 标注数据：首先需要有一组标注好的动漫图像或视频数据，其中每个动漫角色的面部都被准确地标注出来。
- 运行检测算法：使用动漫面部检测算法或模型对这些数据进行检测，得到检测结果。
- 计算指标：根据检测结果和标注数据，计算准确率、召回率、精确率等指标。

例如，假设有一个动漫图像数据集，其中有1000个标注的动漫面部。使用某个动漫面部检测算法进行检测后，算法正确检测到了800个面部，同时还有50个面部被错误地检测为非面部，另外有150个面部未被检测到。那么：

- 准确率 = 正确检测到的面部数量 / 总面部数量 = 800 / 1000 = 0.8
- 召回率 = 正确检测到的面部数量 / 实际存在的面部数量 = 800 / 1000 = 0.8
- 精确率 = 正确检测到的面部数量 / 检测到的总面部数量 = 800 / (800 + 50) = 0.941

```python
def calculate_afd(generated_images_path, save=True):
    """
    Calculate AFD (Anime Face Detection) score for generated images.
    
    :param generated_images_path: Path to the directory containing generated images.
    :return: AFD score (percentage of images detected as anime faces)
    """
    results = yolov8_animeface.predict(generated_images_path, save=save, conf=0.8, iou=0.8, imgsz=64)

    anime_faces_detected = 0
    total_images = len(results)

    for result in results:
        if len(result.boxes) > 0:
            anime_faces_detected += 1

    afd_score = anime_faces_detected / total_images
    return afd_score
```

> 单以当前的YOLOv8预训练模型为例，很可能当前模型只学会了判断两个眼睛的区域是 `face`，但没学会判断三个眼睛图像的不是 `face`，这会导致 `AFD`实际上偏高，所以只能作学习用途

```python
# Calculate and print FID and AFD with optional visualization
yolov8_animeface = YOLO('yolov8x6_animeface.pt')
real_images_path = './faces/faces'  # Replace with the path to real images
fid = calculate_fid(real_images_path, './submission')
afd = calculate_afd('./submission')
print(f'FID: {fid}')
print(f'AFD: {afd}')
```

## Model

### Diffusion Model

![image-20241017001147686](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW06-2022\readme.assets\image-20241017001147686.png)


### Diffusion Model

扩散模型（Diffusion Model）是一种生成模型，通过逐步添加噪声来破坏数据，然后学习如何逐步去噪以恢复原始数据。这种模型在图像生成、音频生成等领域表现出色。下面详细介绍扩散模型的训练、推理过程和生成图像的机制。

#### 训练过程

1. **前向扩散过程（Forward Diffusion Process）**：
   - 从真实数据（如图像）开始，逐步添加高斯噪声，直到数据完全变成噪声。
   - 每一步添加的噪声量是预定义的，通常是一个固定的方差序列。
   - 数学上，前向扩散过程可以表示为：
     $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$
     其中，$x_t$ 是第  $t$ 步的噪声数据， $\beta_t $是每一步的方差参数。
   
2. **反向去噪过程（Reverse Diffusion Process）**：
  
   - 训练的目标是学习一个模型，能够从噪声数据逐步去噪，恢复到原始数据。
   - 反向去噪过程是通过最小化去噪后的数据与真实数据之间的差异来实现的。
   - 数学上，反向去噪过程可以表示为：
     
     $p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
     
     其中，$\mu_\theta $和 $\Sigma_\theta$  是模型参数化的均值和方差。

#### 推理过程

1. **反向去噪过程**：
   - 在推理过程中，模型从完全噪声的数据开始，逐步去噪，直到生成清晰的图像。
   - 每一步去噪都需要模型进行一次前向计算，因此推理过程较慢。
   - 数学上，推理过程可以表示为：
     $x_{t-1} = \mu_\theta(x_t, t) + \epsilon_\theta(x_t, t)$
     其中，$\epsilon_\theta $ 是模型预测的噪声。

#### 生成图像

1. **初始化**：
   - 从高斯噪声开始，即 $x_T \sim \mathcal{N}(0, I) $。

2. **逐步去噪**：
   - 逐步应用反向去噪过程，直到生成清晰的图像。
   - 每一步去噪都需要模型进行前向计算，因此生成图像的速度较慢。



### StyleGAN

StyleGAN（Style-based Generative Adversarial Network）是一种基于样式的生成对抗网络，由NVIDIA的研究团队提出。StyleGAN通过引入样式（Style）的概念，使得生成图像的质量和控制性得到了显著提升。

#### 主要特点

1. **样式控制**：
   - StyleGAN通过样式向量（Style Vector）来控制生成图像的特征，如颜色、纹理、形状等。
   - 样式向量通过一个映射网络（Mapping Network）从随机噪声生成，然后通过样式模块（AdaIN）应用到生成器的不同层。

2. **逐层样式应用**：
   - 样式向量在生成器的不同层中逐层应用，使得不同层次的特征可以独立控制。
   - 这种逐层样式应用使得生成图像的细节和全局特征可以分别调整。

3. **噪声注入**：
   - StyleGAN在生成器的每一层中注入随机噪声，以增加生成图像的多样性和细节。
   - 噪声注入使得生成图像的微小细节更加自然和随机。

4. **混合正则化**：
   - StyleGAN引入了混合正则化（Mixing Regularization），通过混合不同样式向量来生成图像，从而提高生成图像的多样性和稳定性。

#### 网络结构

1. **映射网络（Mapping Network）**：
   - 映射网络是一个全连接网络，将输入的随机噪声向量映射到样式向量。
   - 映射网络的作用是将低维的噪声向量转换为高维的样式向量，以便更好地控制生成图像的特征。

2. **生成器（Generator）**：
   - 生成器由多个卷积层组成，每一层都应用样式向量和噪声。
   - 生成器的每一层通过自适应实例归一化（AdaIN）来应用样式向量，从而控制生成图像的特征。

3. **判别器（Discriminator）**：
   - 判别器是一个卷积神经网络，用于区分生成图像和真实图像。
   - 判别器的作用是提供反馈，帮助生成器生成更逼真的图像。

#### 训练过程

1. **生成器训练**：
   - 生成器的目标是生成尽可能逼真的图像，使得判别器无法区分生成图像和真实图像。
   - 生成器的损失函数是生成图像被判别器误判为真实图像的概率。

2. **判别器训练**：
   - 判别器的目标是尽可能准确地区分生成图像和真实图像。
   - 判别器的损失函数是正确区分生成图像和真实图像的概率。

#### 生成图像

1. **初始化**：
   - 从随机噪声开始，通过映射网络生成样式向量。

2. **逐层生成**：
   - 样式向量逐层应用到生成器的不同层，生成图像的特征逐层叠加。
   - 噪声注入使得生成图像的细节更加自然和随机。

3. **混合正则化**：
   - 通过混合不同样式向量生成图像，提高生成图像的多样性和稳定性。


# Baseline

## Simple

just run sample code

## Medium

**data augmentation+more steps(add noise)**

```python
timesteps = 1000            # Number of steps (adding noise) #medium

class Dataset(Dataset):
    
    ...
    
        #################################
        ## DONE: Data Augmentation ##
        #################################
        
        self.transform = T.Compose([ # Medium
            T.Resize(image_size),
            T.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            T.RandomRotation(10),      # Random rotation
            T.ColorJitter(brightness=0.25, contrast=0.25),  # Slight color adjustments
            T.ToTensor()
        ])
```

可参考[DDPM](https://arxiv.org/abs/2006.11239)

## Strong

**调整超参数+beta_schedule**

- 调整channels, dim_mults

```python
channels = 32             # Numbers of channels of the first layer of CNN  # strong
dim_mults = (1, 2, 4, 8)        # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)
```

- Varience Schedule的不同形式

```python
def linear_beta_schedule(timesteps):
    """
    linear schedule 线性时间表, proposed in original ddpm paper
    DDPM: Canonical diffusion model 典型扩散模型
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    #  在 beta_start 和 beta_end 之间生成一个包含 timesteps 个元素的线性时间表
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s=0.008): # Strong
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    生成扩散模型（如扩散概率模型）中的 beta 值序列，用于扩散模型中的噪声调度
    timesteps：时间步数，即扩散过程中的步数
    s：一个小的偏移量，默认值为 0.008，用于调整余弦函数的形状。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) # 生成时间步序列
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2   # 使用余弦函数计算积累alpha值
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # 根据累积 alpha 值计算得到的，表示相邻两个时间步的 alpha 值的比率
    return torch.clip(betas, 0.0001, 0.9999) # 将 betas 限制在 [0.0001, 0.9999] 的范围内

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    '''
    在扩散模型（如 DDPM）中，动态地从一组调度参数（如噪声衰减系数或其他时间相关的参数）中，
    根据当前时间步 t 提取相应的值，并将其转换为适合进一步计算的形状。
    a: 包含包含调度参数（如噪声系数）的张量，形状为(time steps,)
    t: 时间步索引张量，表示要提取的时间步,(batch size,1)
    x_shape: 提取出的张量最终要匹配的形状
    '''
    b, *_ = t.shape   # get batch size "b"
    out = a.gather(-1, t)  # 从a的最后一个维度（列）上提取t索引对应位置的元素
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 调整为 (b, 1, 1, ..., 1) 这样的形状，其中 1 的个数等于 x_shape 的维度减去 1
```

```python
# Gaussian Diffusion Model
class GaussianDiffusion(nn.Module):  #strong
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        beta_schedule = 'cosine', # linear,cosine,sigmoid
        auto_normalize = True
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        self.model = model
        self.channels = self.model.channels # Strong
        self.image_size = image_size

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
```



## Boss

**StyleGAN**

```python
class StyleGANTrainer(object):
    def __init__(
        self, 
        folder, 
        image_size, 
        *,
        train_batch_size=16, 
        gradient_accumulate_every=1, 
        train_lr=1e-3, 
        train_num_steps=100000, 
        ema_update_every=10, 
        ema_decay=0.995, 
        save_and_sample_every=1000, 
        num_samples=25, 
        results_folder='./results', 
        split_batches=True
    ):
        super().__init__()

        dataloader_config = DataLoaderConfiguration(split_batches=split_batches)
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            mixed_precision='no')
        
        self.image_size = image_size

        # Initialize the generator and discriminator
        self.gen = self.create_generator().cuda()
        self.dis = self.create_discriminator().cuda()
        self.g_optim = torch.optim.Adam(self.gen.parameters(), lr=train_lr, betas=(0.0, 0.99))
        self.d_optim = torch.optim.Adam(self.dis.parameters(), lr=train_lr, betas=(0.0, 0.99))
        
        self.train_num_steps = train_num_steps
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # Initialize the dataset and dataloader
        self.ds = Dataset(folder, image_size)
        self.dl = cycle(DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count()))

        # Initialize the EMA for the generator
        self.ema = EMA(self.gen, beta=ema_decay, update_every=ema_update_every).to(self.device)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        
        self.save_and_sample_every = save_and_sample_every
        self.num_samples = num_samples
        self.step = 0

    def create_generator(self):
        return dnnlib.util.construct_class_by_name(
            class_name='training.networks.Generator',
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=self.image_size,
            img_channels=3
        )

    def create_discriminator(self):
        return dnnlib.util.construct_class_by_name(
            class_name='training.networks.Discriminator',
            c_dim=0,
            img_resolution=self.image_size,
            img_channels=3
        )

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'gen': self.accelerator.get_state_dict(self.gen),
            'dis': self.accelerator.get_state_dict(self.dis),
            'g_optim': self.g_optim.state_dict(),
            'd_optim': self.d_optim.state_dict(),
            'ema': self.ema.state_dict()
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, ckpt):
        data = torch.load(ckpt, map_location=self.device)
        self.gen.load_state_dict(data['gen'])
        self.dis.load_state_dict(data['dis'])
        self.g_optim.load_state_dict(data['g_optim'])
        self.d_optim.load_state_dict(data['d_optim'])
        self.ema.load_state_dict(data['ema'])
        self.step = data['step']

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_g_loss = 0.
                total_d_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    # Get a batch of real images
                    real_images = next(self.dl).to(self.device)
                    
                    # Generate latent vectors
                    latent = torch.randn([self.batch_size, self.gen.z_dim]).cuda()
                    
                    # Generate fake images
                    fake_images = self.gen(latent, None)

                    # Discriminator logits for real and fake images
                    real_logits = self.dis(real_images, None)
                    fake_logits = self.dis(fake_images.detach(), None)

                    # Discriminator loss
                    d_loss = torch.nn.functional.softplus(fake_logits).mean() + torch.nn.functional.softplus(-real_logits).mean()

                    # Update discriminator
                    self.d_optim.zero_grad()
                    self.accelerator.backward(d_loss / self.gradient_accumulate_every)
                    self.d_optim.step()
                    total_d_loss += d_loss.item()

                    # Generator logits for fake images
                    fake_logits = self.dis(fake_images, None)

                    # Generator loss
                    g_loss = torch.nn.functional.softplus(-fake_logits).mean()

                    # Update generator
                    self.g_optim.zero_grad()
                    self.accelerator.backward(g_loss / self.gradient_accumulate_every)
                    self.g_optim.step()
                    total_g_loss += g_loss.item()

                self.ema.update()

                pbar.set_description(f'G loss: {total_g_loss:.4f} D loss: {total_d_loss:.4f}')
                self.step += 1

                if self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map(lambda n: self.ema.ema_model(torch.randn([n, self.gen.z_dim]).cuda(), None), batches))
                    all_images = torch.cat(all_images_list, dim=0)
                    utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(np.sqrt(self.num_samples)))
                    self.save(milestone)
                pbar.update(1)

        print('Training complete')

    def inference(self, num=1000, n_iter=5, output_path='./submission'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with torch.no_grad():
            for i in range(n_iter):
                latent = torch.randn(num // n_iter, self.gen.z_dim).cuda()
                images = self.ema.ema_model(latent, None)
                for j, img in enumerate(images):
                    utils.save_image(img, f'{output_path}/{i * (num // n_iter) + j + 1}.jpg')
                    

```

# Question

## Question 1

**Sample 5 images and show the progressive generation. Then, briefly describe their differences in different time steps.**

**采样5图像，并显示渐进式生成。然后，简要描述它们在不同时间步长的差异。**

```python
class GaussianDiffusion(nn.Module):
    
    ...
    
    # Gradescope – Question 1
    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False, num_samples=5, save_path='./Q1_progressive_generation.png'):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]
        samples = [img[:num_samples]]  # Store initial noisy samples

        x_start = None
        
        ###########################################
        ## TODO: plot the sampling process ##
        ###########################################
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t)
            imgs.append(img)
            if t % (self.num_timesteps // 20) == 0:
                samples.append(img[:num_samples])  # Store samples at specific steps
        
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        self.plot_progressive_generation(samples, len(samples)-1, save_path=save_path)
        return ret
    
    ...
    
    # Gradescope – Question 1
    def plot_progressive_generation(self, samples, num_steps, save_path=None):
        fig, axes = plt.subplots(1, num_steps + 1, figsize=(20, 4))
        for i, sample in enumerate(samples):
            axes[i].imshow(vutils.make_grid(sample, nrow=1, normalize=True).permute(1, 2, 0).cpu().numpy())
            axes[i].axis('off')
            axes[i].set_title(f'Step {i}')
        if save_path:
            plt.savefig(save_path)
        plt.show()
```

![image-20241016235848546](D:\Tunny\Documents\Git\Hungyi-Lee-Machine-Learning-Homework\HW06-2022\readme.assets\image-20241016235848546.png)

去噪过程主要是指从完全噪声的图像开始，通过逐步减少噪声，最终生成一个清晰的图像。去噪过程的简单描述：

1. **初始步骤（噪声）：** 在初始步骤中，图像是纯噪声，此时的图像没有任何结构和可辨识的特征，看起来为随机的像素点。
2. **中间步骤：** 模型通过多个时间步（Timesteps）将噪声逐渐减少，每一步都试图恢复更多的图像信息。
   - 早期阶段，图像中开始出现一些模糊的结构和形状。虽然仍然有很多噪声，但可以看到一些基本轮廓和大致的图像结构。
   - 中期阶段，图像中的细节开始变得更加清晰。面部特征如眼睛、鼻子和嘴巴开始显现，噪声显著减少，图像的主要轮廓和特征逐渐清晰。
3. **最终步骤（完全去噪）：** 在最后的步骤中，噪声被最大程度地去除，图像变清晰。

## Question 2

**Canonical diffusion model (DDPM) is slow during inference, Denoising Diffusion Implicit Model (DDIM) is at least 10 times faster than DDPM during inference, and preserve the qualities.** 

**Please describe the differences of training, inference process, and the generated images of the two models respectively. Briefly explain why DDIM is faster.**

**典型扩散模型（DDPM）在推理过程中速度较慢，而去噪扩散隐式模型（DDIM）在推理过程中速度至少是DDPM的10倍，并且保持了推理的质量。**

**请分别描述两种模型在训练、推理过程和生成图像方面的差异。简要解释为什么DDIM更快。**

在[DDPM/DDIM去噪扩散概率模型和GANs](https://blog.csdn.net/qq_42875127/article/details/141611658?spm=1001.2014.3001.5501)给出详细解答，可参考。

### 典型扩散模型（DDPM）

**训练过程：**

1. **前向扩散过程**：
   - 在训练过程中，首先通过逐步添加高斯噪声将真实图像转换为噪声图像。这个过程称为前向扩散过程。
   - 每一步添加的噪声量是预定义的，通常是一个固定的方差序列。

2. **反向去噪过程**：
   - 训练的目标是学习一个模型，能够从噪声图像逐步去噪，恢复到原始图像。
   - 反向去噪过程是通过最小化去噪后的图像与真实图像之间的差异来实现的。

**推理过程**：

1. **反向去噪过程**：
   - 在推理过程中，模型从完全噪声的图像开始，逐步去噪，直到生成清晰的图像。
   - 每一步去噪都需要模型进行一次前向计算，因此推理过程较慢。

**生成图像：**

- 生成图像的质量依赖于每一步去噪的准确性。
- 由于每一步都需要模型进行前向计算，生成图像的速度较慢。

### 去噪扩散隐式模型（DDIM）

**训练过程：**

1. **前向扩散过程**：
   - 与DDPM类似，DDIM在训练过程中也通过逐步添加高斯噪声将真实图像转换为噪声图像。
   - 不同的是，DDIM在前向扩散过程中引入了隐式变量，使得每一步的噪声添加过程更加灵活。

2. **反向去噪过程**：
   - 训练的目标同样是学习一个模型，能够从噪声图像逐步去噪，恢复到原始图像。
   - 由于引入了隐式变量，DDIM的反向去噪过程更加高效。

**推理过程**：

1. **反向去噪过程**：
   - 在推理过程中，DDIM通过跳过一些中间步骤，直接从噪声图像生成清晰的图像。
   - 由于跳过了中间步骤，DDIM的推理速度显著提高。

**生成图像：**

- 生成图像的质量与DDPM相当，但由于跳过了中间步骤，生成图像的速度显著提高。
- DDIM通过隐式变量和跳步策略，实现了更快的推理速度。

**为什么DDIM更快？**

1. **隐式变量**：
   - DDIM引入了隐式变量，使得每一步的噪声添加过程更加灵活。这使得模型在反向去噪过程中可以跳过一些中间步骤，从而提高推理速度。

2. **跳步策略**：
   - DDIM在推理过程中采用了跳步策略，即直接从噪声图像生成清晰的图像，而不需要每一步都进行前向计算。这种策略显著减少了推理过程中的计算量，从而提高了推理速度。

3. **高效的去噪过程**：
   - 由于隐式变量的引入和跳步策略的应用，DDIM的去噪过程更加高效。模型不需要每一步都进行前向计算，从而减少了推理时间。

**总结**

- **DDPM**：训练和推理过程较为繁琐，每一步都需要模型进行前向计算，生成图像的速度较慢。
- **DDIM**：通过引入隐式变量和跳步策略，显著提高了推理速度，同时保持了生成图像的质量。

DDIM之所以更快，主要是因为它在推理过程中跳过了一些中间步骤，减少了计算量，从而实现了更快的生成速度。
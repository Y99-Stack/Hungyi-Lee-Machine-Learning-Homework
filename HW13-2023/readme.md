[toc]

# task

通过network compression完成图片分类，数据集跟hw3中11种食品分类一致。需要设计小模型student model，参数数目少于60k，训练该模型接近teacher model的精度test-Acc ≅ 0.902

# link

[Kaggle](https://www.kaggle.com/competitions/ml2023spring-hw13)

# baseline

## Simple Baseline 

Just run the sample code

## Medium Baseline

**loss function定义为 KL divergence**，公式如下：
$$
Loss=αT^2×KL(p||q)+(1−α)(Original Cross Entropy Loss),where \ p=softmax(\frac{\text{student's logits}}{T}),and\ q=softmax(\frac{\text{teacher's logits}}{T})
$$
同时**epoch可以增加到50**，其他地方标有#medium的也要修改

```python
# medium
def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.5, temperature=5.0):
    # ------------TODO-------------
    # Refer to the above formula and finish the loss function for knowkedge distillation using KL divergence loss and CE loss.
    # If you have no idea, please take a look at the provided useful link above.
    student_prob = F.softmax(student_logits/temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits/temperature, dim=-1)
    KL_loss = (teacher_prob * (teacher_prob.log() - student_prob)).mean()
    CE_loss = nn.CrossEntropyLoss()(student_logits, labels)
    loss = alpha * temperature**2* KL_loss + (1 - alpha) * CE_loss
    return loss
```

## Strong Baseline

**用depth-wise and point-wise convolutions修改model architecture+增加epoch**

* other useful techniques
    * [group convolution](https://www.researchgate.net/figure/The-transformations-within-a-layer-in-DenseNets-left-and-CondenseNets-at-training-time_fig2_321325862) (Actually, depthwise convolution is a specific type of group convolution)
    * [SqueezeNet](!https://arxiv.org/abs/1602.07360)
    * [MobileNet](!https://arxiv.org/abs/1704.04861)
    * [ShuffleNet](!https://arxiv.org/abs/1707.01083)
    * [Xception](!https://arxiv.org/abs/1610.02357)
    * [GhostNet](!https://arxiv.org/abs/1911.11907)

还可以应用中间层特征学习，这里我没有做

```python
# strong
def dwpw_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels), #depthwise convolution
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, 1), # pointwise convolution
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
```

## Boss Baseline

**Other advanced Knowledge Distillation（FitNet/RKD/DM) + 增加epoch + Depthwise & Pointwise Conv layer（深度可分离卷积）** 

当然也可以应用中间层特征学习

### FitNet Knowledge Distillation

FitNet focuses on transferring knowledge from intermediate feature  representations (hidden layers) instead of just using the output logits. The student model is trained to mimic the feature maps from certain  layers of the teacher model.

```python
#boss
def loss_fn_fitnet(teacher_feature, student_feature, labels, alpha=0.5):
    """
    FitNet Knowledge Distillation Loss Function.
    
    Args:
    - teacher_feature: The feature maps from a hidden layer of the teacher model.
    - student_feature: The feature maps from the corresponding hidden layer of the student model.
    - labels: Ground truth labels for the task.
    - alpha: Weighting factor for the feature distillation loss.
    
    Returns:
    - loss: Combined loss with cross-entropy and feature map alignment.
    """
    # Mean squared error loss to align feature maps of teacher and student
    feature_loss = F.mse_loss(student_feature, teacher_feature)

    # Hard label cross-entropy loss for the student output (classification)
    hard_loss = F.cross_entropy(student_feature, labels)
    
    # Combine both losses
    loss = alpha * hard_loss + (1 - alpha) * feature_loss
    return loss
```

### Relational Knowledge Distillation (RKD)

Relational Knowledge Distillation focuses on transferring the  relationships (distances and angles) between data samples as learned by  the teacher. The student model is trained to match these relationships  instead of just focusing on output probabilities.

```python
# boss
def pairwise_distance(x):
    """Calculate pairwise distance between batch samples."""
    return torch.cdist(x, x, p=2)

def angle_between_pairs(x):
    """Calculate angles between all pairs of points in batch."""
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    norm = diff.norm(dim=-1, p=2, keepdim=True)
    normalized_diff = diff / (norm + 1e-8)
    angles = torch.bmm(normalized_diff, normalized_diff.transpose(1, 2))
    return angles

def loss_fn_rkd(teacher_feature, student_feature, labels, alpha=0.5):
    """
    Relational Knowledge Distillation Loss Function.
    
    Args:
    - teacher_feature: Teacher model feature embeddings.
    - student_feature: Student model feature embeddings.
    - labels: Ground truth labels.
    - alpha: Weighting factor for relational distillation loss.
    
    Returns:
    - loss: Combined relational knowledge and hard label loss.
    """
    
    # Pairwise distances between features in the teacher and student model
    teacher_dist = pairwise_distance(teacher_feature)
    student_dist = pairwise_distance(student_feature)
    
    # Distillation loss using the L2 norm between relational distances
    distance_loss = F.mse_loss(student_dist, teacher_dist)

    # Angle-based loss between teacher and student feature vectors
    teacher_angle = angle_between_pairs(teacher_feature)
    student_angle = angle_between_pairs(student_feature)
    angle_loss = F.mse_loss(student_angle, teacher_angle)
    
    # Hard label cross-entropy loss for the student output
    hard_loss = F.cross_entropy(student_feature, labels)
    
    # Combine the losses
    loss = alpha * hard_loss + (1 - alpha) * (distance_loss + angle_loss)
    return loss
```

### Distance Metric (DM) Knowledge Distillation

Distance Metric distillation focuses on transferring the distance  metric (such as Euclidean distance or cosine similarity) between  instances in the teacher's feature space to the student model.

```python
def loss_fn_dm(teacher_feature, student_feature, labels, alpha=0.5):
    """
    Distance Metric (DM) Knowledge Distillation Loss Function.
    
    Args:
    - teacher_feature: The feature representations from the teacher model.
    - student_feature: The feature representations from the student model.
    - labels: Ground truth labels for the task.
    - alpha: Weighting factor for distance metric loss.
    
    Returns:
    - loss: Combined distance metric loss and cross-entropy loss.
    """
    # Calculate pairwise distance between teacher and student embeddings
    teacher_dist = pairwise_distance(teacher_feature)
    student_dist = pairwise_distance(student_feature)
    
    # Distance metric loss using Mean Squared Error (MSE) loss
    dist_loss = F.mse_loss(student_dist, teacher_dist)
    
    # Hard label cross-entropy loss for the student's output
    hard_loss = F.cross_entropy(student_feature, labels)
    
    # Combine the losses
    loss = alpha * hard_loss + (1 - alpha) * dist_loss
    return loss
```


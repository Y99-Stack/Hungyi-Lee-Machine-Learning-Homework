[toc]

# Link
[Kaggle](https://www.kaggle.com/competitions/ml2023spring-hw15)
# Task: Few-shot Classification
**The Omniglot dataset**
- background set: 30 alphabets
- evaluation set: 20 alphabets
- Problem setup: 5-way 1-shot classification

**Omniglot数据集**
- 背景集:30个字母
- 评估集:20个字母
- 问题设置:5-way 1-shot分类
![](https://i-blog.csdnimg.cn/direct/b140f11fe87040b8b81d5634f8b2e3f8.png)[Definition of support set and query set](https://www.youtube.com/watch?v=PznN0w7dYc0&t=5s)
# Baseline
## Simple—transfer learning
直接把sample code运行即可
- traing: 
对随机选择的5个任务进行正常分类训练验证/测试
- validation / testing:
对五个 Support Images 进行微调，并对Query Images进行推理

Slover首先从训练集中选择5个任务，然后对选择的5个任务进行正常分类训练。在推理中，模型在支持集`support set`图像上微调inner_train_step步骤，然后在查询集`Query Set`图像上进行推理。
为了与元学习Slover保持一致，基本Slover具有与元学习Slover完全相同的输入输出格式

```python
def BaseSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    inner_train_step=1,
    inner_lr=0.4,
    train=True,
    return_labels=False,
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    for meta_batch in x:
        # Get data
        support_set = meta_batch[: n_way * k_shot]
        query_set = meta_batch[n_way * k_shot :]

        if train:
            """ training loop """
            # Use the support set to calculate loss
            labels = create_label(n_way, k_shot).to(device)
            logits = model.forward(support_set)
            loss = criterion(logits, labels)

            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, labels))
        else:
            """ validation / testing loop """
            # First update model with support set images for `inner_train_step` steps
            fast_weights = OrderedDict(model.named_parameters())


            for inner_step in range(inner_train_step):
                # Simply training
                train_label = create_label(n_way, k_shot).to(device)
                logits = model.functional_forward(support_set, fast_weights)
                loss = criterion(logits, train_label)

                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                # Perform SGD
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            if not return_labels:
                """ validation """
                val_label = create_label(n_way, q_query).to(device)

                logits = model.functional_forward(query_set, fast_weights)
                loss = criterion(logits, val_label)
                task_loss.append(loss)
                task_acc.append(calculate_accuracy(logits, val_label))
            else:
                """ testing """
                logits = model.functional_forward(query_set, fast_weights)
                labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    batch_loss = torch.stack(task_loss).mean()
    task_acc = np.mean(task_acc)

    if train:
        # Update model
        model.train()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return batch_loss, task_acc

```
## Medium — FO-MAML
FOMAML（First-Order MAML）是MAML（Model-Agnostic Meta-Learning）的一种简化版本。MAML是一种元学习算法，旨在通过训练模型使其能够在少量新数据上快速适应新任务。FOMAML通过忽略二阶导数来简化MAML的计算过程，从而提高计算效率。它在许多情况下表现良好，尤其是在计算资源有限的情况下。然而，它也可能在某些任务上表现不如完整的MAML。

MAML的核心思想是通过在多个任务上进行训练，使得模型能够在面对新任务时，只需少量数据就能快速收敛到一个好的参数配置。具体来说，MAML的训练过程包括两个层次的优化：

- 内层优化（Inner Loop）：在每个任务上进行少量的梯度更新，以适应该任务。

- 外层优化（Outer Loop）：在所有任务上进行梯度更新，以优化模型的初始参数，使得模型在面对新任务时能够快速适应。

```python
""" Inner Loop Update """
grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=False) # create_graph=False：这个参数表示在计算梯度时不创建计算图。在FOMAML中，我们只关心一阶导数，因此不需要创建计算图
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )


""" Outer Loop Update """
        # TODO: Finish the outer loop update
        # raise NotimplementedError
        meta_batch_loss.backward()
        optimizer.step()
```

## Strong — MAML
```python
""" Inner Loop Update """
grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )


""" Outer Loop Update """
        # TODO: Finish the outer loop update
        # raise NotimplementedError
        meta_batch_loss.backward()
        optimizer.step()
```
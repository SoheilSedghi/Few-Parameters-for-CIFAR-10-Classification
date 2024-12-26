# DNN-with-Few-Parameters-for-CIFAR-10-Classification
Deep Neural Network with Few Parameters for CIFAR-10 Classification

# Main Employed Methods explanation:
* #### **Separable Convolutions**:

Separable convolutions are a powerful technique in deep learning, particularly in computer vision tasks. They offer significant advantages in terms of computational efficiency and model complexity reduction, while often maintaining comparable performance to standard convolutional layers.  In our model, I employ depthwise separable convolutions instead of traditional convolutional blocks.

Depthwise Separable Convolutions: This type of separable convolution divides the standard convolution operation into two steps:
* Depthwise Convolution: Applies a single filter to each input channel independently, capturing spatial features.
*Pointwise Convolution (1x1 Convolution): Combines the output of the depthwise convolution across channels, learning linear combinations of input channels.

The key point of **MobileNet** architecture is its use of depthwise separable convolutions.
<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs13369-024-09131-1/MediaObjects/13369_2024_9131_Fig2_HTML.png" width="400px" alt="Image description">

* #### **Residual Connection**:

The key point of **ResNet** architecture is the introduction of residual connections or skip connections. These connections allow the gradient to flow directly from earlier layers to later layers, mitigating the vanishing gradient problem that often arises in deep neural networks.

<img src="
https://production-media.paperswithcode.com/methods/resnet-e1548261477164.png" width="400px" alt="Image description">


* #### **Scaling Method**:

My model leverages the core principle of *EfficientNet*, which involves compound scaling. This method simultaneously scales the network's depth, width, and resolution to optimize performance. By adopting higher resolution input images of 320x320 pixels and employing a deeper and wider network architecture, I aim to further enhance the model's accuracy and efficiency.

<img src="
https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_10.45.54_PM.png" width="600px" alt="Image description">


# Other Employed Methods Explanation:

* **Data Augmentation**

Data augmentation is a technique used to artificially increase the size of a dataset by creating modified versions of existing images. This technique is particularly useful when working with smaller datasets or when the dataset lacks diversity. By augmenting the data, I can expose the model to a wider range of variations, improving its generalization ability and reducing overfitting.

```
v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),

```
v2.AutoAugmentPolicy.CIFAR10 is a pre-defined AutoAugment policy that has been optimized for the CIFAR-10 dataset. It provides a set of augmentation operations and their probabilities that have been found to be effective for this specific dataset. Using this policy can significantly improve the performance of models trained on CIFAR-10.

* **L2 Regularization**

L2 regularization, also known as weight decay, is a technique used to prevent overfitting in machine learning models. It works by adding a penalty term to the loss function that is proportional to the square of the weights. This penalty term encourages the model to learn smaller weights, which can lead to simpler and more generalizable models. I use a weight decay of 0.005. This means that the L2 regularization term will contribute 0.005 times the sum of the squared weights to the overall loss function.

```
optimizer = torch.optim.SGD( model.parameters(),
                            lr = learning_rate,
                             momentum = 0.9,
                             weight_decay = 0.005,
                             )
```


* **Dropout**

Dropout is a regularization technique used in neural networks to prevent overfitting. During training, a random subset of neurons is randomly dropped out, meaning their activations are set to zero. This forces the network to learn more robust features and reduces the reliance on any specific neuron. I apply dropout layers after fully connected layer. The dropout rate, which determines the probability of a neuron being dropped, is set 0.2.  


* **Cosine Annealing**

The Cosine Annealing Learning Rate Scheduler is a popular technique for adjusting the learning rate during training. It follows a cyclical learning rate schedule that resembles a cosine function.

```
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0.00000001)

```

# Neural-Architecture-Search-Self-Learning
Survey and simple reproduction of NAS, 2023 Spring in THUSZ research group (remote)

# A Survey of Automated Design of Deep Neural Network Architectures - Research Report

# Yifan SHI, Renmin University of China

## Content Structure

This research report provides an explanation of selected research achievements in the field of neural architecture search, divided into four parts. Firstly, I summarize and organize the main content of seven papers, analyzing the strengths and limitations of these research findings. Secondly, I replicate the ENAS and DARTS approaches, both lightweight neural architecture search methods, and successfully discover a CNN Cell architecture for each. I also fine-tune and validate the architectures mentioned in the papers. Furthermore, I outline the process of architecture search and illustrate it with a flowchart. Finally, I compare the weight sharing methods used in current work and analyze their advantages and limitations from various perspectives.

## 1. Literature Review

### 1.1 Overview

![Classification](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/Classification.png)

As shown in the above figure, in the field of Neural Architecture Search (NAS), research methods can mainly be categorized into the following categories based on their approaches: **reinforcement learning-based methods, gradient descent-based methods, and hypernetwork-based methods**. Among them, the gradient descent-based method called DARTS, although utilizing a "hypernetwork" globally, is classified separately as its hypernetwork **only represents the probability distribution of a single model rather than the entire set of models**. In the following sections, based on the content of the papers, I will provide an abstraction and summary of the steps, major advantages, and limitations of these methods.

* **Reinforcement Learning-based methods** (Reinforcement Learning NAS)
  * **Steps**: Neural architecture search is performed using reinforcement learning. A controller generates an architecture structure, which is then trained to convergence. The controller model is adjusted based on feedback, and this process is repeated until convergence is reached.
  * **Major advantages**: Accurate and highly effective.
  * **Major limitations**: High computational resource consumption, not suitable for large datasets, and difficult to determine if the optimum has been reached.
  * **Representative work**: *Neural Architecture Search with Reinforcement Learning (Zoph, et al.)*

* **Gradient Descent-based methods** (Differentiable NAS)
  * **Steps**: The originally discrete search space is transformed into a continuous one. The abstract architecture is encoded as a set of vectors representing selection biases. Mathematical methods are used to compute gradients, and the vectors and weights are updated iteratively to obtain the optimal architecture and weights simultaneously.
  * **Major advantages**: Relatively accurate and fast computation speed.
  * **Major limitations**: Small search space, resulting architectures are often simple and prone to overfitting, and can get trapped in local optima.
  * **Representative work**: *DARTS: Differentiable Architecture Search (Liu, et al.)*

* **Hypernetwork-based methods** (One-Shot NAS)
  * **Steps**:
    * First, a supernet or a set of shared weights is constructed and optimized. Randomly sample one or multiple architectures from the supernet using a certain strategy (such as reinforcement learning, uniform sampling, progressive shrinking, sandwich, etc.). Update the weights of the architecture's network through one forward inference and backward propagation process. This process is repeated until the supernet converges, at which point all models in the supernet are considered optimal.
    * Next, using algorithms such as evolutionary algorithms, random algorithms, or hierarchical filtering algorithms, select an optimal substructure from the supernet based on specific hardware constraints. Start training from scratch or use it directly (without post-processing) to obtain the final optimal model.
  * **Major advantages**: Accurate, large search space, and fast search speed (when hardware adaptability is not considered) or good adaptability (when hardware adaptability is considered).
  * **Major limitations**: Weak interpretability, difficulty in training the supernet while achieving hardware adaptability, low supernet utilization when hardware adaptability is not implemented.
  * **Representative works**: *ENAS: Efficient Neural Architecture Search via Parameter Sharing (Pham, et al.)*; *Single Path One-Shot Neural Architecture Search with Uniform Sampling (Stamoulis, et al.)*; *Once-for-All: Train One Network and Specialize it for Efficient Deployment (Cai, et al.)*; *BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models (Bender, et al.)*

The subsequent content in this section provides summaries and analyses of the **main content, advantages, and limitations** of the seven papers individually.

### 1.2 Neural Architecture Search with Reinforcement Learning

#### Summary

This paper introduces a method for Neural Architecture Search (NAS) using reinforcement learning (RL). Traditional methods for architecture design are manually designed, which often require significant prior experience and time, resulting in low efficiency. Additionally, although many sophisticated structures have been designed, it is difficult to mathematically prove their optimality due to the intricate nature of their components. This paper proposes an automated architecture search method using reinforcement learning.

The paper utilizes an RNN network as the controller, which generates an architecture structure for an RNN cell. The generated architecture is then trained to convergence, and its accuracy on the validation set is evaluated. The accuracy serves as a reward for reinforcement learning, and the controller is updated accordingly.

This method can generate simple CNN architectures, including features related to convolutional kernels and parameters for convolution operations. The following diagram illustrates an example:

![截屏2023-04-01 17.52.18](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/%E6%88%AA%E5%B1%8F2023-04-01%2017.52.18.png)

As the method requires **gradient-based parameter updates**, it adopts reinforcement learning methods proposed in *Ronald J. Williams' Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Machine Learning, 1992)*. Based on experience, the method simplifies and optimizes the variance using mathematical techniques. The gradient of the controller's parameters is given as:

$$
\frac 1 m \sum_{k=1}^m \sum_{t=1}^T \nabla_{\theta_c}\log P(a_t|a_{(t-1):1};\theta_c)(R_k-b)
$$

To accelerate training and search processes, distributed machine learning techniques are employed. To increase the complexity and expand the search space of the generated architectures, the controller incorporates **residual network connections (skip connections)** and anchor nodes to indicate whether such connections are present.

![截屏2023-04-01 17.44.39](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/%E6%88%AA%E5%B1%8F2023-04-01%2017.44.39.png)

The above diagram provides an example of generating an RNN cell architecture from the controller's output. The overall architecture is defined by a $k$-ary tree, with each tree node having two attributes representing the predicted operation and activation function. The final two special nodes complete the cell structure. In the paper's implementation, an 8-ary tree is used.

The authors trained 12,800 CNN architectures on the CIFAR-10 dataset using a massive amount of computational resources (800 GPUs) and trained an RNN cell architecture on the Penn Treebank dataset for 35 epochs. The method achieved excellent results.

#### Analysis of Advantages and Limitations

The **advantages** of this paper are:

* It uses reinforcement learning to automate the design of a solution for a specific computer vision problem.
* It explores a **relatively diverse range of architectures**, with a rich search space that includes CNNs and RNN cell architectures.
* It achieves **promising results**.

The **limitations** of this paper are:

* It is computationally **inefficient**, requiring significant computational resources and making it **infeasible for large-scale dataset training**.
* The **CNN architectures** could be improved further, as the method does not consider methods such as dilated convolutions or grouped convolutions.
* The **RNN architectures** could be improved further, as

 the current approach focuses only on cell sampling.

* The method exhibits **limited dataset adaptability**, performing well on specific datasets.

### 1.3 DARTS: Differentiable Architecture Search

#### Summary

This paper introduces the Differentiable Architecture Search (DARTS) method, which transforms the discrete architecture search problem into a continuous optimization problem by utilizing the relaxation technique. It represents the discrete operations as continuous weight vectors that can be optimized using gradient descent, eliminating the need for reinforcement learning.

The encoding scheme used in DARTS involves applying the softmax activation function to transform a specific operation into a mixture of softmax probabilities over all possible operations. This allows a set of continuous weight vectors, denoted as $\alpha^{(i,j)}$, to represent the "preference" for each operation between node $i$ and node $j$. The preference, denoted as $\overline{o}^{(i,j)}$, depends only on the continuous vector $\alpha^{(i,j)}$, as the set of operations $\mathcal{O}$ is predetermined. The preference is calculated as follows:

$$
\overline{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}}\exp(\alpha_{o'}^{(i,j)})}
$$

When generating the actual architecture, the "most probable operation" is selected instead of the preference. In other words:

$$
o^{(i,j)} = \arg\max_{o \in \mathcal{O}} \alpha_o^{(i,j)}
$$

The architecture search problem is then formulated as a two-level optimization problem: minimizing the validation error of the final architecture generated by the encoding vector and training the architecture generated by the encoding vector on the training set until convergence. The weights $w$ and the architecture encoding vector $\alpha$ are trained in an alternating manner until convergence.

Mathematical approximations are employed to approximate the gradients of the architecture encoding vector:

* First, the gradient of the encoding vector at the optimal weights is approximated as the gradient of the encoding vector corresponding to the weights after one optimization step.
* Second, the differentiation operation, which involves complex matrix operations, is approximated as finite differences.
* [Optional]: The second-order gradients are approximated as zero, considering only first-order approximations.

These approximations reduce the overall complexity to an acceptable level.

The paper conducts CNN searches on the CIFAR-10 dataset and RNN searches on the Penn Treebank dataset, achieving better results on CIFAR-10 than the Efficient Neural Architecture Search (ENAS) method. The DARTS method is also successfully transferred to large-scale datasets such as ImageNet, demonstrating the portability of the trained model architectures.

#### Analysis of Advantages and Limitations

The **advantages** of this paper are:

* It proposes a novel and efficient method that **transforms discrete search into continuous search**.
* The search space is more flexible, allowing for the search of arbitrary shapes and sizes of subgraphs.
* The paper presents optimizations for **gradient computation**.
* It requires **less computational resources**.
* The searched architectures demonstrate **good performance**.

The **limitations** of this paper are:

* The generated architectures tend to be **relatively simple**, as the search space is not as large.
* Due to the non-convexity and complexity of the search problem, the method is susceptible to **overfitting and sensitivity to random seeds**.
* It still requires **non-negligible computational resources** compared to other methods like ENAS.
* The search space needs to be specified in advance (e.g., the set of possible operations $\mathcal{O}$).

### 1.4 Efficient Neural Architecture Search via Parameter Sharing

#### Summary

In this paper, the research team identifies that the slow computation in traditional discrete neural architecture search (NAS) is mainly due to the training of each child model to convergence, only to measure its accuracy and discard all the trained weights. The authors propose the strategy of **parameter sharing** and design the Efficient Neural Architecture Search (ENAS) method based on this idea. ENAS achieves a speedup of up to 1000 times compared to NAS, while producing comparable results.

First, the entire search space is defined as a complete directed acyclic graph (DAG) denoted as $G=\{V,E\}$. Each node in the graph corresponds to an operation, and the data flow is represented by directed edges. A subset of edges $E_0 \subseteq E$ is sampled from this graph, resulting in a subgraph $G_0 = G[E_0]$. This subgraph represents an architecture sampled during the neural architecture search.

The controller is still an RNN. When generating the architecture of an RNN cell, predictions are made for each node to determine the following attributes:

* Which input edge to select.
* What operation should be performed at this node.

For nodes without outgoing edges, the average of their values is taken as the output of the cell. For edge $e \in E_0:i \rightarrow j$, it represents the computation $h_j = \text{Op}_j (h_i \cdot \mathbf{W_{j,i}^{(h)}})$. The weight matrix $\mathbf{W}$ is **shared** across the entire network during training.

During the training process, the controller parameters $\theta$ and the weights $w$ of the child network are trained in an alternating manner. First, the controller is fixed, and using a Monte Carlo method, multiple child networks are sampled based on a sampling policy specified by $\theta$. The gradients of $w$ are computed for these networks, and the Supernet is updated synchronously. It is found that computing gradients based on a single sampled child network is sufficient. Next, the weights $w$ are fixed, and the gradients of $\theta$ are computed using the method described in Section 1.2. This process is iterated multiple times until convergence, at which point the Supernet is considered fully trained.

After that, the **Supernet-related parameters** are fixed, and the fine-tuning step begins. Multiple initial child models are sampled, and the best one is selected for training from scratch to obtain the best child model and its weights.

For CNN architecture search, similar to the RNN architecture design, decisions are made for each node: which previous nodes should be used as inputs and what operations to perform. The paper also introduces a method to search for individual CNN cells (layers) to compose the final CNN architecture, which significantly reduces the search space compared to previous CNN architecture search methods.

ENAS achieves good performance. It is tested on the CIFAR-10 and Penn Treebank datasets, where it performs slightly worse than traditional NAS (Section 1.2), but is 1000 times faster.

#### Analysis of Advantages and Limitations

The **advantages** of this research are as follows:

* It achieves **extremely high efficiency** by utilizing **parameter sharing**, making it 1000 times faster than traditional NAS.
* It achieves **good performance** comparable to traditional NAS.
* The search space is **fully explored**, as the Supernet is a complete graph.

The **limitations** of this research are as follows:

* The effectiveness is not tested on **large-scale datasets** such as ImageNet.
* The **large number of weights** in the Sup

ernet, as each edge in the complete graph has a corresponding weight matrix.

* The **limited expressiveness** of the Supernet, as each node can only represent a few different architectures.
* It **does not satisfy constraints** that may be present in the desired architectures.

### 1.5 Searching for MobileNetV3

This paper focuses on the search for the MobileNetV3 architecture and does not primarily address neural architecture search (NAS) itself. It introduces modified weight factors to adapt to the latency variations in small models. Therefore, I won't go into further details about this paper here.

### 1.6 Single Path One-Shot Neural Architecture Search with Uniform Sampling

#### Summary

Traditional gradient-based methods for neural architecture search (NAS) often suffer from getting stuck in local optima and may not find the global optimum. The choice of initial random seeds has a strong impact on the results, and it is still unclear which sampling strategies are effective. In this paper, a NAS method using **uniform sampling** is proposed to address these issues by introducing randomness. The method adopts a "Single Path One-Shot" strategy, which significantly reduces the search cost. Experimental results on multiple datasets demonstrate that this method achieves comparable performance to state-of-the-art NAS methods.

Recent research has used the "One-Shot" approach, which generates paths in the supernet by randomly dropping some edges, effectively training all parameters in the graph uniformly and potentially finding the global optimum. However, this approach is highly sensitive to the dropout rate, making training challenging. The "Single Path" approach used in this paper overcomes this issue.

Unlike traditional supernets, the proposed supernet in this paper aims to optimize all architectures in the search space simultaneously. The objective is to find an optimal **shared weight** $W_\mathcal A$, such that the expected loss of the network $\mathcal N(a, W(a))$ sampled according to architecture $a$ from the search space $\mathcal A$ is minimized. The optimization is expressed by the following equation:
$$
W_\mathcal A=\mathop{\text{argmin}}_W\  \mathop {\mathbb{E}}_{a\sim \Gamma(\mathcal A)}[\mathcal L_{train}(\mathcal{N}(a,W(a)))]
$$
The search space is also expanded by introducing strategies for **channel number search** and **mixed-precision quantization search**, allowing architecture search to adapt to multi-channel search and mapping bit-widths, thereby increasing the complexity of the models.

Since the architectures generated by the "One-Shot" approach are random and there is no direct way to select the best architecture for training, an **evolutionary search** strategy is employed. Each generated network $\mathcal N(a, W(a))$ is only evaluated through inference to obtain its accuracy on the validation set. An evolutionary algorithm is then used to search for the best architecture, which is trained from scratch.

#### Analysis of Advantages and Limitations

The **advantages** of this research are as follows:

* It achieves **high efficiency and performance**.
* It has **better theoretical performance** by reducing the likelihood of getting stuck in local optima through one-shot training and uniform sampling.
* The search space is further expanded to include multi-channel and mixed-precision searches, increasing model complexity.
* It can **satisfy certain constraints**.
* The evolutionary search strategy improves the quality of the searched architectures.

The **limitations** of this research are as follows:

* It can only generate CNN networks that include convolutional layers and fully connected layers.
* The sampling strategy is uniform, which **still imposes some range limitations**.
* The One-Shot strategy makes it difficult for the supernet to converge, and the **time required to fully train the supernet to convergence is high** compared to methods like ENAS that use alternating training. Training for multiple epochs does not guarantee complete convergence of the supernet.

### 1.7 Once-for-All: Train One Network and Specialize it for Efficient Deployment

#### Summary

Traditional NAS methods require computing and searching from scratch when deploying a deep learning model on different devices, which incurs significant overhead. This paper proposes a method called Once-for-All (OFA), where the supernet is trained only once and can be efficiently deployed on multiple platforms. OFA models searched for edge devices exhibit higher efficiency or higher accuracy.

OFA aims to generate CNN architectures that can have arbitrary layer depths (elastic depth), channel numbers (elastic width), kernel sizes (elastic kernel size), and image resolutions (elastic resolution). This significantly expands the search space.

During the training of the supernet, there are two basic approaches:

* Enumerate all subnetworks and compute gradients precisely at each step: **unacceptable time complexity**.
* Sample a few subnetworks and estimate gradients at each step: **inaccurate and difficult to converge**.

This paper proposes a method called **Progressive Shrinking** to train the supernet. It starts by training a large model (Full Model) and progressively shrinks the model by adjusting different dimensions (depth, width, kernel size, resolution). The training simultaneously involves the large model and the shrinking models (Shrinking Model), enabling the supernet to support models of various scales. The large and small models have a **nested relationship**, allowing the small models to **share the core weight parameters** from the large model, which accelerates the training process.

The paper explains the **weight sharing strategies** for depth, width, and kernel size:

* Different kernel sizes are shared using **transformation matrices**.
* Different depths are shared by **pruning** the first $D$ layers and using them for weight sharing.
* Different widths are shared by applying a **sorting algorithm** to identify the most important channels for weight sharing.

Finally, after training an OFA supernet, the paper measures the performance and efficiency metrics (accuracy-latency twin) corresponding to different architecture scales (Depth-Width-KernelSize). This mapping is used to create a look-up table that guides subsequent architecture search using the OFA supernet:
$$
\sigma:[D,W,K]\rightarrow[ACC,Latency]
$$

#### Analysis of Advantages and Limitations

The research has the following **notable advantages**:

* It proposes a method, OFA Supernet, that can **flexibly support models of different scales (hardware conditions) without retraining**: the training cost of the supernet can be amortized across various applications.
* Small models can be directly derived from the supernet with **little or no post-processing** required.
* The proposed Progressive Shrinking method allows accurate and rapid training of the supernet.
* The method has been tested on **different devices (including edge devices)** and shows good performance and efficiency metrics.

The **limitations** of this research are as follows:

* The supernet acts as a **black box** for subsequent applications, making it **difficult to interpret and fine-tune** if problems arise after training.
* The supernet is trained using a Full Model, and all subsequent submodels continue training using the weights from the previous stage. This may lead to getting stuck in **local optima** (although it achieves good results in practice).
* The performance measurement look-up table may incur **significant overhead** or require sacrificing a large portion of the search space.
* The supernet **cannot be easily adapted to different datasets and problem domains**, requiring significant resources for retraining if a change is needed.
* The weight sharing strategies for different scales in the supernet training process may still be somewhat **unscientific**, and they cannot achieve unbiased estimation in theory (although they yield good results in practice).

### 1.8 BigNAS: Scaling up Neural Architecture Search with Big Single-Stage Models

#### Summary

Traditional Supernet-based NAS methods involve two stages: training the supernet and further adjusting a model generated by the super network using post-processing. However, the post-processing step can be time-consuming. BigNAS aims to propose a **one-step** approach to directly generate usable models.

The overall method consists of two steps:

- First, **train a large-scale super network** that can directly generate a deployable small model through slicing or other methods.
- Second, when given certain constraints, use a **coarse-to-fine-grained filtering** approach to obtain the optimal model.

In contrast to the Once-For-All method mentioned in 1.7, BigNAS adopts the approach of **simultaneously training all submodels**. During each training iteration, it uses a **sandwich method** to sample the largest, smallest, and $N$ randomly sized models, calculates the overall gradient, and updates its own weights.

BigNAS uses **in-place distillation**, where the loss for the largest model is computed using ground truth, while the losses for other models are computed using the soft labels predicted by the largest model. This method ensures consistency among all models, leading to a slight improvement in performance.

Additionally, it is observed that the convergence speed differs between large and small models. Large models tend to overfit while small models have not yet converged. To address this, BigNAS replaces the learning rate with **exponentially decaying with a constant ending**. This allows the large model to oscillate near the optimal solution without overfitting, while accelerating the convergence of the small model.

Finally, the task is to select the optimal model that satisfies the given constraints. First, several **reference scale parameters** are provided. Then, based on the models generated using these parameters, the optimal scale parameters that meet the conditions are identified. Finally, these parameters are **randomly mutated** to obtain the final optimal model.

#### Analysis of Advantages and Limitations

This research is similar to the Once-For-All approach.

The **advantages** of this research are:

* Training a single-stage super network that can **directly generate the required models without additional training**.
* Simultaneously training different-sized optimal models using an optimization approach.

The **limitations** of this research are:

* BigNAS relies on certain hyperparameters (such as the learning rate constant), which can be **difficult to tune**.
* Although the goal of BigNAS is to achieve optimal models without post-processing, in practice, **additional training may still be required** in certain scenarios.
* The coarse-to-fine-grained filtering still requires **significant computational time**, which can be considered a form of post-processing.

## 2. Replication Work

I have studied and researched the repositories (if available) associated with each paper. After analyzing the hardware requirements for replicating each paper, I have chosen to **attempt replication of ENAS and DARTS** (with the lowest computational resource requirements) due to limitations on the duration of server usage for replication purposes.

The server environment used for the replication work is as follows:

* OS: Ubuntu 20.04.5 LTS x86_64
* CPU: 12th Gen Intel i5-12600K (12) @ 3.600GHz
* GPU: NVIDIA GeForce RTX 2080 Ti Rev. A (12GB)
* Memory: 32GB
* CUDA Version: 10.0
* cuDNN Version: 7.4.2

### 2.1 Replication of ENAS

In the original paper's GitHub repository, I obtained the source code provided by the research team. The replication was performed using the following environment:

* Python Version: 2.7
* Tensorflow Version: 1.13

I used the exact same versions of TensorFlow and Python as mentioned in the paper but was unable to run the code for neural architecture search on the CIFAR-10 dataset with the predefined parameters. Despite multiple attempts to fix the issue, I was unsuccessful, possibly due to insufficient server memory (which may require at least 64GB of RAM). Therefore, I made modifications to the `data_util.py` file by deleting the fifth training batch of CIFAR-10 and reducing the training data size. I performed a **re-run of the search with custom parameters**, conducting 150 epochs of search.

```python
  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    # "data_batch_5",   Removed the last batch
  ]
```

After approximately 5 hours of training, I obtained the following results. The accuracy on the validation set was 61.4%, and on the test set, it was 59.75%.

```python
[0 4 0 1 1 3 1 0 0 1 3 0 1 1 0 2 0 1 1 3]
[0 0 1 0 1 1 1 1 2 4 1 1 4 1 1 1 1 2 1 0]
val_acc=0.6250
--------------------------------------------------------------------------------
Epoch 150: Eval
Eval at 32850
valid_accuracy: 0.6140
Eval at 32850
test_accuracy: 0.5975
```

The research team provided an explanation that a micro unit with `B + 2` blocks can be specified using `B` blocks, corresponding to blocks numbered `2, 3, ..., B+1`. Each block is represented by four numbers:

```python
index_1, op_1, index_2, op_2
```

Here, `index_1` and `index_2` can be any previous index, and `op_1` and `op_2` can take values `[0, 1, 2, 3, 4]`, corresponding to `separable_conv_3x3`, `separable_conv_5x5`, `average_pooling`, `max_pooling`, and `identity` operations, respectively.

The research team also provided the pre-searched optimal architecture using a similar approach:

```python
fixed_arc="0 2 0 0 0 4 0 1 0 4 1 1 1 0 0 1 0 2 1 1 1 0 1 0 0 3 0 2 1 1 3 1 1 0 0 4 0 3 1 1"
```

Using a visualization tool, I visualized the CNN Cell architecture that I searched for and the one pre-searched by the research team, as shown in figures (a) and (b) below. It can be observed that the two architectures have some similarities, but due to the removal of a portion of the training dataset and adjustment of certain parameters, the results are not identical.

![截屏2023-04-04 00.37.57](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/%E6%88%AA%E5%B1%8F2023-04-04%2000.37.57.png)

Finally, I performed additional fine-tuning on the architecture pre-searched by the research team using the `cifar-micro-search` approach mentioned in the paper.

```python
epoch=629   ch_step=218900 loss=0.002567 lr=0.0001   |g|=0.5652   tr_acc=144/144 mins=1664.85   
epoch=629   ch_step=218950 loss=0.013373 lr=0.0001   |g|=4.6466   tr_acc=143/144 mins=1665.21   
epoch=629   ch_step=219000 loss=0.001038 lr=0.0001   |g|=0.1008   tr_acc=144/144 mins=1665.56   
epoch=629   ch_step=219050 loss=0.001635 lr=0.0001   |g|=0.4037   tr_acc=144/144 mins=1665.92   
epoch=629   ch_step=219100 loss=0.000772 lr=0.0001   |g|=0.1711   tr_acc=144/144 mins=1666.28   
epoch=629   ch_step=219150 loss=0.002294 lr=0.0001   |g|=0.6714   tr_acc=144/144 mins=1666.63   
epoch=629   ch_step=219200 loss=0.001445 lr=0.0001   |g|=0.5475   tr_acc=144/144 mins=1666.99   
Epoch 630: Eval
Eval at 219240
test_accuracy: 0.9619
```

As shown above, after approximately 30 hours of training with 630 epochs, the model achieved an accuracy of 96.19%, which corresponds to a classification error of 3.81%. This result is close to the described classification error of 3.54% mentioned in the paper (as shown in the figure below).

![截屏2023-04-02 23.24.01](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/%E6%88%AA%E5%B1%8F2023-04-02%2023.24.01.png)

### 2.2 DARTS Reproduction

In the original paper's GitHub repository, I obtained the publicly available source code from the research team. I performed the reproduction using the following environment:

* Python Version: 3.6
* Pytorch Version: 0.3.1
* Torchvision Version: 0.2.0

```c
2023-04-03 08:37:30,882 train 000 4.491183e-02 98.437500 100.000000
2023-04-03 08:39:06,123 train 050 1.870327e-02 99.571078 100.000000
2023-04-03 08:40:41,354 train 100 1.855421e-02 99.644183 100.000000
2023-04-03 08:42:16,606 train 150 1.920131e-02 99.565397 100.000000
2023-04-03 08:43:51,818 train 200 1.885082e-02 99.611318 100.000000
2023-04-03 08:45:27,087 train 250 1.907752e-02 99.638944 100.000000
2023-04-03 08:47:02,286 train 300 1.911234e-02 99.652201 100.000000
2023-04-03 08:48:37,542 train 350 1.870171e-02 99.675036 100.000000
2023-04-03 08:49:53,315 train_acc 99.676000
2023-04-03 08:49:53,449 valid 000 4.773718e-01 87.500000 100.000000
2023-04-03 08:49:57,625 valid 050 3.327558e-01 90.318627 99.571078
2023-04-03 08:50:01,801 valid 100 3.695028e-01 89.402847 99.504950
2023-04-03 08:50:05,977 valid 150 3.672881e-01 89.372930 99.492964
2023-04-03 08:50:10,154 valid 200 3.796697e-01 89.132463 99.448072
2023-04-03 08:50:14,330 valid 250 3.788103e-01 89.162102 99.477092
2023-04-03 08:50:18,507 valid 300 3.811134e-01 89.192276 99.454942
2023-04-03 08:50:22,684 valid 350 3.814137e-01 89.271724 99.483618
2023-04-03 08:50:25,993 valid_acc 89.264000
```

First, I used the **second-order approximation** method mentioned in the paper to search for a model with **8 layers and an initial channel size of 16** on the CIFAR-10 dataset. I performed the search for 50 epochs, and after approximately 6 hours of searching, I obtained the results shown in the figure above.

 This model performed exceptionally well on the training set with an accuracy of 99.676%, but only achieved an accuracy of 89.264% on the validation set. This indicates that DARTS-based methods have a risk of **overfitting**.

The specific parameters of this model are as follows:

```python
genotype = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
```

Using the visualization tool provided by the research team, the CNN cell architecture obtained from the above search and the architecture pre-searched by the research team can be visualized as shown in the figure below. Figures **(a)** and **(b)** show the Normal Cells obtained by me and the research team, respectively, which are similar in nature. Figures **(c)** and **(d)** show the Reduction Cells obtained by the research team and me, respectively. It should be noted that due to the inclusion of random operations during the computation process and slight differences in our initial parameters, there may be some variations in the results.

![截屏2023-04-04 00.50.33](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/%E6%88%AA%E5%B1%8F2023-04-04%2000.50.33.png)

Next, I ran the pre-trained model provided by the research team. It achieved an accuracy of 97.37% on the test set of CIFAR-10, corresponding to an error rate of 2.63%. This aligns with the reported performance of 2.76% ± 0.09% in the paper.

```c
Files already downloaded and verified
04/03 11:56:30 PM test 000 1.233736e-01 96.875000 100.000000
04/03 11:56:34 PM test 050 1.105459e-01 97.120095 99.959150
04/03 11:56:37 PM test 100 1.074739e-01 97.359733 99.948432
04/03 11:56:37 PM test_acc 97.369997
```

![截屏2023-04-04 17.15.48](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/%E6%88%AA%E5%B1%8F2023-04-04%2017.15.48.png)

## 3. Overview of Architecture Search Processes

For the three categories of NAS methods mentioned in section 1.1 (reinforcement learning-based methods, gradient-based methods, and supernet-based methods), each has its own unique process. The following flowcharts illustrate these processes.

### 3.1 Reinforcement Learning-based Methods

![RL.drawio](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/RL.drawio.png)

For reinforcement learning-based methods, the NAS process is relatively straightforward and involves an iterative process. The controller generates architectures based on a policy and trains them until convergence. Then, the models are evaluated, and feedback (reward) is provided to the controller to update the generation policy. The process continues until convergence is reached, resulting in the optimal model.

### 3.2 Gradient-based Methods

![FlowGraph.drawio (1)](https://raw.githubusercontent.com/Stone1011/oss/master/uPic/FlowGraph.drawio%20(1).png)

The flowchart in the figure above **(a)** illustrates the workflow of gradient-based methods:

1. The original discrete search space is transformed into a continuous one by encoding the abstract architecture into a set of vectors that represent the selection preferences.
2. A code $\alpha$ (i.e., a distribution) is sampled, and the closest network $\mathcal{N}$ to that distribution is obtained by decoding the code (re-discretization).
3. The loss $\mathcal{L}$ and gradients ($\nabla_w$ and $\nabla_\alpha$) of the network are computed, and the weights $w$ and architecture code vector $\alpha$ are updated.
4. If convergence is not reached, repeat the previous step.
5. Obtain the optimal architecture.

### 3.3 Supernet-based Methods

The flowchart in the figure above **(b)** illustrates the workflow of supernet-based methods, which can be divided into two stages: **training the supernet** (1-4) and **searching for the optimal architecture** (5-7):

1. Define a supernet, usually represented by a directed acyclic graph $G$, which represents the entire search space.
2. Sample one or multiple subnetworks $G_0 \subset G$ based on a certain strategy (randomly or based on previous results) and form one or multiple architectures $\alpha$.
3. Compute the loss of the architecture and update the weights of the $G_0$ portion in $G$.
4. If convergence is not reached, repeat the previous two steps to obtain a well-trained supernet.
5. Use a specific algorithm to search for the optimal substructure that meets certain constraints.
6. Further training of the substructure from scratch may be required based on the specific situation.
7. Obtain the optimal architecture.

## 4 Analysis of Weight Sharing Strategies

In the papers mentioned, the following weight sharing strategies are discussed:

* **DARTS**: *DARTS: Differentiable Architecture Search (Liu, et al.)*
* **ENAS**: *ENAS: Efficient Neural Architecture Search via Parameter Sharing (Pham, et al.)*
* **SPOS**: *Single Path One-Shot Neural Architecture Search with Uniform Sampling (Stamoulis, et al.)*
* **OFA**: *Once-for-All: Train One Network and Specialize it for Efficient Deployment (Cai, et al.)*
* **BigNAS**: *BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models (Bender, et al.)*

Based on the explanations and summaries in the first section, I have drawn the following conclusions. In the subsequent explanations, I will use the bolded names (DARTS, ENAS, SPOS, OFA, BigNAS) to refer to the research outcomes of the aforementioned papers. Additionally, for the sake of fair comparison, I will focus on the differences in the search approaches for **CNN network architectures**.

### 4.1 Weight Sharing Strategies and Search Space

The gradient-based method proposed by **DARTS** (via Continuous Relaxation) is a *generalized weight sharing strategy*. It uses a super-network to represent the distribution of model architectures, rather than the entire set of models. Multiple sub-networks are not carved out. Since this super-network has only one architecture initialization process, the weights are shared among the sub-architectures throughout the entire iteration. The search space is not large and requires predefining all possible cases as the preset space.

**ENAS** uses a series of weight matrices $\mathbf{W_{j,i}^{(h)}}$ to represent the weights multiplied by the data flowing from node $i$ to node $j$ in the super-network $G$. The search space in this approach is a pre-defined super-network $G$. In each iteration, the controller samples a subgraph $G_0 \subset G$ based on a certain strategy $\theta$ to form the architecture $\alpha$ for that iteration. Then, the weights $w$ and the strategy $\theta$ are updated based on the specific loss $\mathcal{L}_{train}(\mathcal N(\alpha, w))$ of the network.

The super-network in **SPOS** is a large hierarchical structure where each node (choice block) represents the selection of multiple operations of the same structure. A single path in the entire super-network represents a final architecture. Each operation choice in this method corresponds to a weight, which is shared among all sampled architectures from the super-network. The search space is large, and each node can have multiple sampling choices, while each convolutional layer can have multiple channel and quantization precision choices.

The super-network in **OFA** is an even larger hierarchical structure where each node has different choices for depth, width, and kernel size. Each node has a shared weight, and the shared weights of smaller structures are obtained by cutting or transforming the weights of larger structures. The sampling strategy used is Progressive Shrinking, which gradually trains architectures of smaller scales.

The super-network in **BigNAS** is similar to **OFA**, but it uses the Sandwich sampling strategy, where multiple architectures of different sizes are trained each time, and the weights of smaller architectures are obtained by cutting the weights of larger architectures. Additionally, BigNAS aims to directly obtain the final model without any additional training.

### 4.2 Comparison of Advantages and Limitations

#### Search Space

As mentioned in the previous section, given similar super-network sizes, the ordering

 of the search space sizes for the five methods is as follows:
$$
\mathbf{DARTS}\approx \mathbf{ENAS} < \mathbf{SPOS} <\mathbf{OFA} \approx \mathbf{BigNAS}
$$

* **DARTS and ENAS** share weights only under predefined conditions.
* **SPOS** shares weights across several randomly generated channel and quantization precision conditions in the machine.
* **OFA and BigNAS** can cut each weight into different choices for depth, width, and kernel size.

As a result, the number of polymorphic variations for each weight gradually increases, leading to an increasing search space (**more inclined towards increased support for micro-hardware environments rather than macro-architectural enlargement and complexity**).

#### Accuracy

After considering the trade-off between accuracy and search space, it is not necessarily better to have a higher number of polymorphic variations for individual weights. A preference for hardware environment polymorphism **impairs the expressive power of individual weights**, making weight convergence more challenging and potentially deviating from the optimal values for specific hardware conditions.

Therefore, in subsequent research, achieving the same effect may result in significantly larger Supernet sizes and longer training times (especially for **OFA and BigNAS**). However, as they are "Once-For-All" or can accommodate numerous constraints, this is considered a reasonable cost.

#### Computational Time

The super-network designs of **DARTS, ENAS, and SPOS** aim to **accelerate computation through weight sharing**, making them highly efficient. However, they have very weak or no support for different constraint conditions.

On the other hand, **OFA and BigNAS** are designed to **realize a super-network that can be reused for multiple hardware conditions** to avoid redundant computation. Although the computation cost of a single super-network is high, it can be amortized through subsequent reuse.

#### Memory Consumption

Since the **DARTS** super-network represents only one model, the entire network parameters need to be loaded into memory (GPU memory) during computation, which is disadvantageous when the Supernet sizes are similar. However, the Supernet size of **DARTS** is generally not excessively large.

Due to training only one sampled subgraph at a time, **ENAS and SPOS** have very low memory overhead.

**OFA and BigNAS** simultaneously train several sampled subgraphs at the same time, which may result in larger memory consumption. However, considering that the devices used to train such Supernets are typically powerful, this is not a major limitation.

#### Sensitivity to Hyperparameters

**DARTS** is more sensitive to hyperparameters because it uses gradient-based methods, which can only find local optima. The initial architecture selection is crucial.

**ENAS and BigNAS** have hyperparameters that significantly affect the results, such as the predefined search space and learning rate constants.

**SPOS and OFA** have higher randomness in their processes, and according to the papers, they do not rely heavily on specific hyperparameters.

#### Mathematical Interpretability

**DARTS** utilizes methods with a solid mathematical foundation, ensuring that it can find local optima under appropriate parameter conditions.

The effectiveness of weight sharing in other methods is difficult to explain.

## Conclusion

After conducting research on this topic, I have gained a preliminary understanding of the basic methods of NAS. Through reading code, attempting reproductions, and analyzing multiple directions of NAS work, I have developed a further understanding of the historical development of NAS and its future trends.

In summary, NASes evolving from the initial state of being "**single-model, resource-intensive, low interpretability, and sensitive**" towards becoming "**multi-hardware-condition, low average resource consumption, high interpretability, and strong stability**." This trend will continue to guide future research.

## References

[1] Cai H, Gan C, Wang T, et al. Once-for-all: Train one network and specialize it for efficient deployment[J]. arXiv preprint arXiv:1908.09791, 2019.

[2] Cai H, Zhu L, Han S. Proxylessnas: Direct neural architecture search on target task and hardware[J]. arXiv preprint arXiv:1812.00332, 2018.

[3] Guo Z, Zhang X, Mu H, et al. Single path one-shot neural architecture search with uniform sampling[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16. Springer International Publishing, 2020: 544-560.

[4] Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 1314-1324.

[5] Krizhevsky A, Hinton G. Learning multiple layers of features from tiny images[J]. 2009.

[6] Liu H, Simonyan K, Yang Y. Darts: Differentiable architecture search[J]. arXiv preprint arXiv:1806.09055, 2018.

[7] Pham H, Guan M, Zoph B, et al. Efficient neural architecture search via parameters sharing[C]//International conference on machine learning. PMLR, 2018: 4095-4104.

[8] Tan M, Chen B, Pang R, et al. Mnasnet: Platform-aware neural architecture search for mobile[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 2820-2828.

[9] Williams R J. Simple statistical gradient-following algorithms for connectionist reinforcement learning[J]. Reinforcement learning, 1992: 5-32.

[10] Yu J, Jin P, Liu H, et al. Bignas: Scaling up neural architecture search with big single-stage models[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16. Springer International Publishing, 2020: 702-717.

[11] Zoph B, Le Q V. Neural architecture search with reinforcement learning[J]. arXiv preprint arXiv:1611.01578, 2016.

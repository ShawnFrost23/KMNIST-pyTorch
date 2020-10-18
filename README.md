  ### KMNIST Data Set
  
  * We will be implementing networks to recognize handwritten Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or KMNIST for short. The paper describing the dataset is available here. It is worth reading, but in short: significant changes occurred to the language when Japan reformed their education system in 1868, and the majority of Japanese today cannot read texts published over 150 years ago. This paper presents a dataset of handwritten, labeled examples of this old-style script (Kuzushiji). Along with this dataset, however, they also provide a much simpler one, containing 10 Hiragana characters with 7000 samples per class. This is the dataset we will be using.


  ### Neural Network Varients

  * 1st Nueral network is a single linear layer with log softmax tuning.

  * 2nd Nueral network is 2 layered. The hidden layers have Tanh tuning while the output has log softmax.

  * 3rd Neural network is 4 layered. 2 convolutional layers follwed by a hidden layer which are activated using reLU and a final output layer activated by log softmax.

  ### How to run

  * Single Layer Nueral network can be ran using 'python3 kuzu_main.py --net lin'

  * 2 layer neural network can be ran using 'python3 kuzu_main.py --net full'

  * 2 convolutional layers with one hidden layer and 1 output layer can be ran using 'python3 kuzu_main.py --net conv'

  ### Requrired libraries

  * Only pyTorch 1.6.0 is needed to run the code. It can be installed using 'pip3 install pytorch' or 'conda install pytorch' depending on your preference.


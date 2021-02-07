# Mitigating Catastrophic Forgetting in LSTM-RNNs and Biologically Constrained RNNs

For my Masters' Thesis, I employed Deep Generative Replay to Mitigate Catastrophic Forgetting in LSTMs and Biologically Constrained RNNs. Deep Generative Replay is a framework for Mitigating Catastrophic Forgetting in Neural Networks where acccess to past data may not be available. It involves training a Generative Model to learn to generate pseudo samples of old data that resemble real samples of old data. Samples generated by this Generative Model are then replayed interspersed with new data, when the original neural network is trained to learn the next task. This is inspired by how humans and other mammals are able to acquire new knowledge while retaining previously learnt information. The Generative Model(GAN) and the LSTM were built in Pytorch, while the Biologically Inspired RNN was written in Theanos. The Biologically Constrained RNN used here is a heavily adapted version of the network used in [Pycog], but with massive modifications, be it to the kind of data the network can accept as input, to the optimisation algorithm itself. The code for the Biological RNN can be found in the ['pycog-master' folder](pycog-master), while the code for the GAN and the LSTM can be found in the files ['MainCode1.ipynb'](MainCode1.ipynb) and 'Thesis 2.ipynb'. This thesis was Supervised and Approved by Professor Zhe Chen at NYU's Department of Neuroscience and Professor Davi Geiger at NYU's Courant Institute.





## Features:

![Features](Images/features.jpeg)




## Architecture for the CNN-LSTM:

![CNNLSTM Arch](Images/cnnlstm.jpeg)



## Results:

### Machine Learning Metrics:

![ml metrics](Images/ml_metrics.jpeg)

### Financial Metrics

![finance](Images/financial_metrics.jpeg)

We also trained the Multi Layer Perceptron(MLP) for 3-class classification, changing the targets such that each could be classified to have a rate of return in the top 10%, the bottom 10% or the middle for the day. One can see that some Financial metrics, like Sharpe Ratio, Max Drawdown and Returns improve significantly.

![finance](Images/3classification.jpeg)


## Report:

[Report](https://github.com/amartyap/Predicting-Stock-Returns-PTSA-Project/blob/master/Project%20Report_PTSA.pdf)


## Team:
1. Amartya Prasad 
2. Steffen Roehershieim          
3. Tianmu Zhao 

## Referencing:

Please reference this work, if you decide to use it.

# transformer_presentation

## Reading:
 "Sequence to Sequence Learning with Neural Networks" by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le
 
 **Link**: https://arxiv.org/pdf/1409.3215.pdf

## Introduction and Overview
The paper addresses the problem of machine translation, which involves translating a sentence from one language to another. Rather than using RNN model, The authors propose a new approach that uses a large deep LSTM neural network to directly translate sentences.

### Sequence Learning Challenge
- DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality
- Many problems, such as speech recognition and machine translation, are best expressed with sequences whose lengths are not known a-priori.
- Deep Neural Networks are hard to handle certain senario when the length of the input and output sequences is not known beforehand

## Modeling and Structure

**DNN VS. LSTM** 

**DNN**

Recurrent neural networks is a natural generalization of feedforward neural networks to sequences. Given a sequence of inputs, a standard RNN computes a sequence of outputs by iterating the following equation

<img width="556" alt="Screenshot 2023-03-25 at 10 57 39 AM" src="https://user-images.githubusercontent.com/89152255/227728236-a7b81ced-0eb3-449d-9bb1-21a70ad93e06.png">
Advantage:

The RNN can map sequences to sequences when the alignment between the inputs the outputs is known. For sequence to sequence learning, one strategy of general sequence learning is to map the input sequence to a fixed-sized vector using one RNN, and then to map the vector to the target sequence with another RNN.

Disadvantage:

While RNN could work in principle as RNN is provided with all the relevant information, it would be difficult to train the RNNs due to the resulting long term dependencies.

**LSTM**
The authors chose to use LSTMs as LSTMS are better at handling long-term dependencies. The LSTM estimates the conditional probability p(y1, . . . , yT ′ |x1, . . . , xT ) where (x1,...,xT)is an input sequence and (y1,...,yT′) is its corresponding output sequence whose length T′ may differ from T . 

<img width="757" alt="Screenshot 2023-03-25 at 12 23 51 PM" src="https://user-images.githubusercontent.com/89152255/227732240-f5069f44-f946-42a3-aaaf-639c08736144.png">

Advantage:

The LSTM computes this conditional probability by first obtaining the fixed-dimensional representation v of the input sequence (x1 , . . . , xT ) given by the last hidden state of the LSTM, and then computing the probability of y1, . . . , yT ′ with a standard LSTM-LM formulation whose initial hidden state is set to the representation v of x1, . . . , xT



**Approach and Imporvement:** 



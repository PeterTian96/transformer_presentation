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

### **DNN VS. LSTM** 

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

### **Model** 

**Model Architecture**

With LSTMs' advantages, The idea is to use one LSTM to read the input sequence, one timestep at a time, to obtain large fixed- dimensional vector representation, and then to use another LSTM to extract the output sequence from that vector. The second LSTM is essentially a recurrent neural network language model except that it is conditioned on the input sequence. The LSTM’s ability to successfully learn on data with long range temporal dependencies makes it a natural choice for this application due to the considerable time lag between the inputs and their corresponding output

<img width="1268" alt="Screenshot 2023-03-25 at 12 38 30 PM" src="https://user-images.githubusercontent.com/89152255/227732909-2fb7546f-fbf0-48eb-b00e-3548f501f199.png">

In this sample, the model reads an input sentence “ABC” and produces “WXYZ” as the output sentence. The model stops making predictions after outputting the end-of-sentence token. Note that the LSTM reads the input sentence in reverse, because doing so introduces many short term dependencies in the data that make the optimization problem much easier.

**Question1: Why LSTMs are good in translation learning?**

A useful property of the LSTM is that it learns to map an input sentence of variable length into a fixed-dimensional vector representation. Given that translations tend to be paraphrases of the source sentences, the translation objective encourages the LSTM to find sentence representations that capture their meaning, as sentences with similar meanings are close to each other while different sentences meanings will be far. 


### **Approach and Imporvement:** 

**Dataset:** The WMT’14 English to French dataset

**Training Data** Using the datset, There are 12M senences consisting of 348M French words and 304M English words are generated

**Decoding and Rescore:** The goal is to train LSTM model by maximizing the log probability of a correct translation T given the source sentence S. Here is the training object:

<img width="441" alt="Screenshot 2023-03-25 at 2 11 10 PM" src="https://user-images.githubusercontent.com/89152255/227736784-e8a29341-30fa-4939-a791-1b521dac1798.png">

Once training is complete, the next step is to produce translations by finding the most likely translation according to the LSTM:

<img width="332" alt="Screenshot 2023-03-25 at 2 12 11 PM" src="https://user-images.githubusercontent.com/89152255/227736813-0d3f01ee-fd23-46d6-baa4-f1315d39377b.png">

**Reversing the Source Sentences**:

During the training process, It is cleared that the LSTM learns much better when the source sentences are reversed. It is possible  the presence of many short term dependencies in the dataset causes this phenomenon. When concatenating a source sentence with a target sentence, each word in the source sentence is often far from its corresponding word in the target sentence, resulting in a large "minimal time lag". **However, reversing the words in the source sentence preserves the average distance between corresponding words in the source and target language.**

**Question2: Why LSTMs are good in Long sentence learning as well?**

To achieve good results on long sentences, the order of words in the source sentence was reversed while the target sentences remained unchanged in both the training and test sets. **This resulted in the introduction of many short-term dependencies that simplified the optimization problem, enabling the LSTMs to effectively handle long sentences.** The key technical contribution of this work is the simple yet effective approach of reversing the words in the source sentence.

**Training details**:

We used deep LSTMs with 4 layers, with 1000 cells at each layer and 1000 dimensional word embeddings, with an input vocabulary of 160,000 and an output vocabulary of 80,000. The resulting LSTM has 384M parameters of which 64M are pure recurrent connections. Parallelization is also used in the training process

<img width="419" alt="Screenshot 2023-03-25 at 2 25 32 PM" src="https://user-images.githubusercontent.com/89152255/227737382-472e78c1-a1ae-47e8-ab99-8299fe4a40c4.png">

**Result**:

<img width="928" alt="Screenshot 2023-03-25 at 2 29 56 PM" src="https://user-images.githubusercontent.com/89152255/227737583-b026a2a0-bfcf-448d-92fa-fb77f3c6c252.png">

The primary result of the study is that a BLEU score of 34.81 was achieved on the WMT'14 English to French translation task by extracting translations directly from an ensemble of 5 deep LSTMs.This is by far the best result achieved by direct translation with large neural networks. For comparison, the BLEU score of an SMT baseline on this dataset is 33.30

<img width="1091" alt="Screenshot 2023-03-25 at 2 32 36 PM" src="https://user-images.githubusercontent.com/89152255/227737722-ca9693fd-a6ed-4691-b1cd-a6bb4d90876e.png">

The left plot shows the performance of the system as a function of sentence length, where the x-axis corresponds to the test sentences sorted by their length and is marked by the actual sequence lengths. There is no degradation on sentences with less than 35 words, there is only a minor degradation on the longest sentences. The right plot shows the LSTM’s performance on sentences with progressively more rare words, where the x-axis corresponds to the test sentences sorted by their “average word frequency rank”

## Critical Thinking
1. **Incomplete Explanation of the Model's Improvement:**

Although the authors offer a hypothesis on why reversing the source sentences led to improved LSTM performance, they fail to provide a comprehensive explanation. Thus, further investigation is necessary to enhance the model's effectiveness. Additionally, it remains unclear whether the reversal of source sentences impacted the performance of the DNN.

2. **Limitation in Training Dataset:**

A fixed vocabulary for both languages was used in the model, simplifying the process but potentially leading to inaccuracies in translation. This limitation does not accurately reflect the full range of words used in the languages being translated, making it imperative to improve the model's accuracy.

3. **Need for More Studies:**

The model's impressive results on a specific translation task cannot guarantee its performance on other sequence-to-sequence problems. It is crucial to explore whether the model can handle out-of-vocabulary words or different languages with varying syntax and grammar rules.


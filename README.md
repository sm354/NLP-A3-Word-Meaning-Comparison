# Word Meaning Comparison

Words often have multiple meanings associated with them and the surrounding context often determines the specific meaning of the word which is used. 

For example, consider the following sentences (S1, S2, S3)

1. We often used to play on the <u>bank</u> of the river 
2. We lived along the <u>bank</u> of the Ganges. 
3. He cashed a check at the <u>bank</u> 

S1 and S2 use the same meaning of the word bank (river bed) while S1 and S3 use different meanings of the word bank (river bed vs. financial institution).

In this work we develop deep neural networks to solve this classification problem ie whether a word has same or different meaning in different sentences. This is part of [Natural Language Processing](https://www.cse.iitd.ac.in/~mausam/courses/col772/autumn2021/) course taken by [Prof Mausam](https://www.cse.iitd.ac.in/~mausam/).

## Results

We evaluate our model's performance using the [Word-in-Context (WiC) Dataset](https://pilehvar.github.io/wic/). WiC is a reliable benchmark for the evaluation of context-sensitive word embeddings.

|        Model         | Validation Accuracy (%) | Test Accuracy (%) |
| :------------------: | :---------------------: | :---------------: |
| LSTM (with word2vec) |          63.5           |       55.5        |
|         BERT         |           71            |       69.8        |

#### Takeaways

- Using libraries like TorchText, hugging face, and PyTorch for building and training nlp deep neural networks. 
- Fine tuning pre-trained models (available in hugging face) using Trainer API or without any API. Lack of torchtext documentation makes it extremely difficult to use, but it does help in simplifying pre-processing stage. 
- In part-A we use torchtext, but in part-B we avoid it for easy manipulations as per our design choices. 
- Documentation of hugging face is more than complete i.e. the functionalities contain too many parameters and arguments to understand. Thus, working with hugging face requires time and efforts but it is justified since the models itself are too big! 
- From implementation perspective (i.e. the ease of implementing our ideas and manipulating the functionalities) we find that APIs like Trainer (in hugging-face), Torchtext are too restrictive and/or difficult to tweak and play with. Using the APIs also pose (additional and irrelevant) challenges in changing the designs of the model architectures. We find it is best to go old school and use data-loaders and iteratively train (or finetune) the models with an appropriate optimizer. 

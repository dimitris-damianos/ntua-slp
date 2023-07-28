import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()
        num_emb, size_emb = embeddings.shape
        # 1 - define the embedding layer
        self.embeddings = nn.Embedding(num_embeddings=num_emb,embedding_dim=size_emb,)

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # EX4
        self.embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})

        # 3 - define if the embedding layer will be frozen or finetuned
        # EX4
        self.embeddings.weight.requires_grad = trainable_emb 

        # 4 - define a non-linear transformation of the representations
        # EX5
        hidden_size = 64
        self.layer = nn.Linear(in_features=size_emb,out_features=hidden_size)
        self.relu = nn.ReLU()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.out_layer = nn.Linear(in_features=hidden_size,out_features=output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embeddings(x)  # EX6
        batch_size,max_length,emb_dim = np.shape(embeddings)

        # 2 - construct a sentence representation out of the word embeddings
        ## each sentence represantation is the mean of its embeddings  
        representations = torch.zeros(size=[batch_size, emb_dim])  # EX6
        for i in range(batch_size):# EX6
            representations[i]=torch.sum(embeddings[i],dim=0)/lengths[i]

        # 3 - transform the representations to new ones.
        representations = self.relu(self.layer(representations))

        # 4 - project the representations to classes using a linear layer
        logits = self.out_layer(representations)  # EX6

        return logits

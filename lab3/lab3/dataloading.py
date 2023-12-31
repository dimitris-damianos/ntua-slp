from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from spacy.lang.en import English
import numpy as np

DATASET = "MR"  ## options: MR, Semeval2017A

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        ## EX2
        ## Tokenize Semeval2017A using TweetTokenizer
        if DATASET == 'Semeval2017A':
            self.data = [TweetTokenizer().tokenize(example) for example in X]
        ## Tokenise MR using scipy.tokenize.Tokenizer
        elif DATASET == "MR":
            self.data=[example.split() for example in X] ## this might change
        else:
            raise ValueError("Not implemented dataset. Pick from MR, Semeval2017A")
        ## find max length of samples in dataset --> required got getitem()
        self.max_length = 0
        for sample in self.data:
            if len(sample)>self.max_length:
                self.max_length = len(sample)
        #print('Tokenized input:',self.data[0])
        self.labels = y
        self.word2idx = word2idx

        

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        example = np.zeros(self.max_length,dtype=np.int64)  ## init example vector
        sample = self.data[index]
        for i in range(len(sample)):  ## change example according to word2idx 
            if sample[i] in self.word2idx:
                example[i] = self.word2idx[sample[i]]
            else:
                example[i] = self.word2idx["<unk>"]
        length = len(self.data[index])
        label = self.labels[index]

        return example, label, length

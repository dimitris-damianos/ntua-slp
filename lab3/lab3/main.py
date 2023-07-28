import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN,LSTM
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

import training, early_stopper, attention


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = True
BATCH_SIZE = 128
EPOCHS = 10
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################


# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

print("Before label encoding:",y_test[:5])

# convert data labels from strings to integers
y_train = LabelEncoder().fit_transform(y_train)  # EX1
y_test = LabelEncoder().fit_transform(y_test)  # EX1
n_classes = (LabelEncoder().fit(y_train)).classes_.size  # EX1 - LabelEncoder.classes_.size

print("After label encoding:",y_test[:5])

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)


## test SentenceDataset
file = open("test-dataset.txt",'w')
for i in range(5):
    example,label,length = test_set[i]
    file.write(f"Item {i}:\nexample={example}\nlabel={label}\nlength={length}\n\n")
file.close()

# EX4 - Define our PyTorch-based DataLoader
#train_loader = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True) # EX7
test_loader = DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=True)  # EX7


## Question 2.1:
## split train to train and validate set
train_loader, val_loader = training.torch_train_val_split(
    dataset=train_set,
    batch_eval=BATCH_SIZE,
    batch_train=BATCH_SIZE
)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
"""model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)"""

### Define LSTM - Question 2
model= LSTM(output_size=n_classes,
             embeddings=embeddings,
             bidirectional=True)

## Import early stopper
from early_stopper import EarlyStopper

early_stopper = EarlyStopper(model=model,
                             save_path='./best-model',
                             patience=5)

# move the mode weight to cpu or gpu
model.to(DEVICE)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()  # EX8

## which layer should we optimize? for layer1 only: model.layer1.parameters()
parameters = model.parameters() # EX8
optimizer = torch.optim.Adam(params=parameters,lr=1e-3)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
total_train = []
total_test = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    
    val_loss, (y_val_pred,y_val_gold) = eval_dataset(val_loader,
                                                     model,
                                                     criterion)
    
    total_train.append(train_loss)
    total_test.append(test_loss)

    ### calculate accuracy,f1score,recall
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )

    y_train_gold = []
    (tmp, _) = y_train_true.shape
    for i in range(tmp):

        if y_train_true[i][0] > y_train_true[i][1]:
            y_train_gold.append(0)
        else:
            y_train_gold.append(1)

    y_test_gold = []
    (tmp, _) = y_test_true.shape
    for i in range(tmp):

        if y_test_true[i][0] > y_test_true[i][1]:
            y_test_gold.append(0)
        else:
            y_test_gold.append(1)

    if early_stopper.early_stop(val_loss): break

print("\n### Train set metrics:\n")
print(f"Accuracy:{accuracy_score(y_train_gold, y_train_pred)}\nF1 score:{f1_score(y_train_gold, y_train_pred, average='macro')}")
print(f"Recall:{recall_score(y_train_gold, y_train_pred, average='macro')}\n")

print("### Test set metrics:\n")
print(f"Accuracy:{accuracy_score(y_test_gold, y_test_pred)}\nF1 score:{f1_score(y_test_gold, y_test_pred, average='macro')}")
print(f"Recall:{recall_score(y_test_gold, y_test_pred, average='macro')}\n")

###########################################################################
# Plot results
###########################################################################
fig,axis = plt.subplots(1,2)
epochs = [i+1 for i in range(len(total_train))]
axis[0].plot(epochs,total_train)
axis[0].set_title("Train loss ")
axis[1].plot(epochs,total_test)
axis[1].set_title("Test loss ")
plt.show()




print("=====" * 5)
print("Simple Self Attention Model")
print("=====" * 5)
model = attention.SimpleSelfAttentionModel(n_classes, embeddings)

# move the mode weight to cpu or gpu
model.to(DEVICE)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()  # EX8

## which layer should we optimize? for layer1 only: model.layer1.parameters()
parameters = model.parameters() # EX8
optimizer = torch.optim.Adam(params=parameters,lr=1e-3)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
total_train = []
total_test = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, True)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion, True)

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion, True)
    
    val_loss, (y_val_pred,y_val_gold) = eval_dataset(val_loader,
                                                     model,
                                                     criterion, True)
    
    total_train.append(train_loss)
    total_test.append(test_loss)

    ### calculate accuracy,f1score,recall
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )

    y_train_gold = []
    (tmp, _) = y_train_true.shape
    for i in range(tmp):

        if y_train_true[i][0] > y_train_true[i][1]:
            y_train_gold.append(0)
        else:
            y_train_gold.append(1)

    y_test_gold = []
    (tmp, _) = y_test_true.shape
    for i in range(tmp):

        if y_test_true[i][0] > y_test_true[i][1]:
            y_test_gold.append(0)
        else:
            y_test_gold.append(1)

    if early_stopper.early_stop(val_loss): break

print("\n### Train set metrics:\n")
print(f"Accuracy:{accuracy_score(y_train_gold, y_train_pred)}\nF1 score:{f1_score(y_train_gold, y_train_pred, average='macro')}")
print(f"Recall:{recall_score(y_train_gold, y_train_pred, average='macro')}\n")

print("### Test set metrics:\n")
print(f"Accuracy:{accuracy_score(y_test_gold, y_test_pred)}\nF1 score:{f1_score(y_test_gold, y_test_pred, average='macro')}")
print(f"Recall:{recall_score(y_test_gold, y_test_pred, average='macro')}\n")


###########################################################################
# Plot results
###########################################################################

fig,axis = plt.subplots(1,2)
epochs = [i+1 for i in range(len(total_train))]
axis[0].plot(epochs,total_train)
axis[0].set_title("Train loss ")
axis[1].plot(epochs,total_test)
axis[1].set_title("Test loss ")
plt.show()

print("=====" * 5)
print("Multi Head Attention Model")
print("=====" * 5)
model = attention.MultiHeadAttentionModel(n_classes, embeddings)

if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()  

## which layer should we optimize? for layer1 only: model.layer1.parameters()
parameters = model.parameters() # EX8
optimizer = torch.optim.Adam(params=parameters,lr=1e-3)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
total_train = []
total_test = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, True)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion, True)

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion, True)
    
    val_loss, (y_val_pred,y_val_gold) = eval_dataset(val_loader,
                                                     model,
                                                     criterion, True)
    
    total_train.append(train_loss)
    total_test.append(test_loss)

    ### calculate accuracy,f1score,recall
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )

    y_train_gold = []
    (tmp, _) = y_train_true.shape
    for i in range(tmp):

        if y_train_true[i][0] > y_train_true[i][1]:
            y_train_gold.append(0)
        else:
            y_train_gold.append(1)

    y_test_gold = []
    (tmp, _) = y_test_true.shape
    for i in range(tmp):

        if y_test_true[i][0] > y_test_true[i][1]:
            y_test_gold.append(0)
        else:
            y_test_gold.append(1)

    if early_stopper.early_stop(val_loss): break


print("\n### Train set metrics:\n")
print(f"Accuracy:{accuracy_score(y_train_gold, y_train_pred)}\nF1 score:{f1_score(y_train_gold, y_train_pred, average='macro')}")
print(f"Recall:{recall_score(y_train_gold, y_train_pred, average='macro')}\n")

print("### Test set metrics:\n")
print(f"Accuracy:{accuracy_score(y_test_gold, y_test_pred)}\nF1 score:{f1_score(y_test_gold, y_test_pred, average='macro')}")
print(f"Recall:{recall_score(y_test_gold, y_test_pred, average='macro')}\n")
###########################################################################
# Plot results
###########################################################################

fig,axis = plt.subplots(1,2)
epochs = [i+1 for i in range(len(total_train))]
axis[0].plot(epochs,total_train)
axis[0].set_title("Train loss ")
axis[1].plot(epochs,total_test)
axis[1].set_title("Test loss ")
plt.show()

print("=====" * 5)
print("Transformer Encoder Model")
print("=====" * 5)
model = attention.TransformerEncoderModel(n_classes, embeddings)

if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()  

## which layer should we optimize? for layer1 only: model.layer1.parameters()
parameters = model.parameters() # EX8
optimizer = torch.optim.Adam(params=parameters,lr=1e-3)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
total_train = []
total_test = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, True)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion, True)

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion, True)
    
    val_loss, (y_val_pred,y_val_gold) = eval_dataset(val_loader,
                                                     model,
                                                     criterion, True)
    
    total_train.append(train_loss)
    total_test.append(test_loss)

    ### calculate accuracy,f1score,recall
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )

    y_train_gold = []
    (tmp, _) = y_train_true.shape
    for i in range(tmp):

        if y_train_true[i][0] > y_train_true[i][1]:
            y_train_gold.append(0)
        else:
            y_train_gold.append(1)

    y_test_gold = []
    (tmp, _) = y_test_true.shape
    for i in range(tmp):

        if y_test_true[i][0] > y_test_true[i][1]:
            y_test_gold.append(0)
        else:
            y_test_gold.append(1)

    if early_stopper.early_stop(val_loss): break

print("\n### Train set metrics:\n")
print(f"Accuracy:{accuracy_score(y_train_gold, y_train_pred)}\nF1 score:{f1_score(y_train_gold, y_train_pred, average='macro')}")
print(f"Recall:{recall_score(y_train_gold, y_train_pred, average='macro')}\n")

print("### Test set metrics:\n")
print(f"Accuracy:{accuracy_score(y_test_gold, y_test_pred)}\nF1 score:{f1_score(y_test_gold, y_test_pred, average='macro')}")
print(f"Recall:{recall_score(y_test_gold, y_test_pred, average='macro')}\n")
###########################################################################
# Plot results
###########################################################################

fig,axis = plt.subplots(1,2)
epochs = [i+1 for i in range(len(total_train))]
axis[0].plot(epochs,total_train)
axis[0].set_title("Train loss ")
axis[1].plot(epochs,total_test)
axis[1].set_title("Test loss")
plt.show()
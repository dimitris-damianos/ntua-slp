from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

#DATASET = 'MR'
DATASET = 'Semeval2017A'

#PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'
#PRETRAINED_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
PRETRAINED_MODEL = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'


LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'lxyuan/distilbert-base-multilingual-cased-sentiments-student': {
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',        
    }
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    # define a proper pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL)

    y_pred = []
    for x in tqdm(X_test):
        # TODO: Main-lab-Q6 - get the label using the defined pipeline 
        label = sentiment_pipeline(x)[0]["label"]
        # The following if statement is used just to prevent errors for MR dataset
        # It classifies as positive, every sentence predicted as neutral
        if DATASET == 'MR' and PRETRAINED_MODEL == 'cardiffnlp/twitter-roberta-base-sentiment' and label == 'LABEL_1':
            label = 'LABEL_2'
        y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])

    y_pred = le.transform(y_pred)
    print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')

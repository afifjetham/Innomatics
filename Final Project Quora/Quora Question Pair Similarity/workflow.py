import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# pip install gensim==4.2.0
from gensim.models import Word2Vec
import regex as re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import contractions
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import mlflow

def load_data(path: str, unwanted_cols: list) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_data(data: pd.DataFrame, target_col: str, test_size: int) -> any:
    X = data.drop([target_col], axis=1)
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= 42)
    return X_train, X_test, y_train, y_test


def preprocessing(raw_text: pd.DataFrame, flag: str ) -> pd.Series:

    # initializing lemmatizer and stemmer 
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()
     
     # Removing special characters and digits
  sentence = re.sub("[^a-zA-Z]", " ", raw_text)
    
  # change sentence to lower case
  sentence = sentence.lower()

  # remove html tags
  #sentence=re.compile(r'<[^>]+>').sub('', sentence)

  # Expanding contractions
    
  expanded_words = []   
  for word in sentence.split():
    expanded_words.append(contractions.fix(word))  
   
    sentence = ' '.join(expanded_words)
    
    
  # remove stop words                
  clean_tokens = [t for t in sentence.split() if not t in stopwords.words("english")]
  
  # Stemming/Lemmatization and POS tagging
  if(flag == 'stem'):
    token_list=[]
    for word, tag in pos_tag(clean_tokens):
      wntag = tag[0].lower()
      wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    
      stem = stemmer.stem(word, wntag) if wntag else word
      token_list.append(stem)

  else:
    token_list=[]
    for word, tag in pos_tag(clean_tokens):
      wntag = tag[0].lower()
      wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None   # a-adjective, r-adverb, n-forms of nouns, v-forms of verbs
    
      lemma = lemmatizer.lemmatize(word, wntag) if wntag else word
      token_list.append(lemma)
     
  return pd.Series([" ".join(token_list),len(token_list)])


def add(a:pd.DataFrame, b:pd.DataFrame) -> pd.DataFrame:
    concatenated = pd.concat([a, b], axis=1)
    
    return concatenated

def q1_q2_comb(a: pd.Series, b: pd.Series) -> pd.Series:
    comb = a + ' '+ b
    return comb

def q1_q2_comb_token(a: pd.Series) ->pd.Series:
    a = [i.split() for i in a]
    return a

# def word2vec(tokened_data: pd.Series, vector_size: int, min_count: int) -> any:
#     word_vec = Word2Vec(list(tokened_data), vector_size=vector_size, min_count= min_count)
#     return word_vec

def document_vector(doc: any, keyed_vectors: any) -> list:
    """Remove out-of-vocabulary words. Create document vectors by averaging word vectors."""
    vocab_tokens = [word for word in doc if word in keyed_vectors.index_to_key]
    if len(vocab_tokens)!=0:
        return np.mean(keyed_vectors.__getitem__(vocab_tokens), axis=0)
    
    else:
        return np.zeros(25)

def train_model(X_tr:np.array, y_tr:pd.Series, estimator: any ) -> any:
    mlflow.set_tracking_uri('sqlite:///mlflow_quora.db')
    mlflow.set_experiment('Quora Questions Pair Similarity')

    with mlflow.start_run():

        lr = estimator
        lr.fit(X_tr, y_tr)
        y_tr_pred = lr.predict(X_tr)

        # mlflow.log_p
        mlflow.set_tag('Dev', "Indramani")
        mlflow.set_tag('Algo', 'Logistic Regression')
        mlflow.log_metric('accuracy', accuracy_score(y_tr, y_tr_pred))
        mlflow.sklearn.log_model('lr_reg', artifact_path = 'models')
        return lr
    


# workflow

def main(path:str):
    VECTOR_SIZE=25
    MIN_COUNT=1
    FLAG = 'lemma'
    path= path
    TARGET_DATA = 'is_duplicate'
    UNWANTED_DATA = ['qid1', 'qid2', 'id']
    TEST_SIZE = 0.3
    # ESTIMATOR = LogisticRegression()
    
    # load dataset
    dataframe = load_data(path=path, unwanted_cols= UNWANTED_DATA)
    
    # drop unwanted cols
    input_data = dataframe.drop(UNWANTED_DATA, axis=1)

    # take sample to perform the task quickly so that we can check the errors quickly
    # samp = input_data.sample(frac= 0.02)

    # test_train_split
    X_train, X_test, y_train, y_test = split_data(input_data, target_col= TARGET_DATA, test_size= TEST_SIZE)

    # text preprocessing

    # text preprocessing for X_train que1 
    temp_df = X_train.question1.apply(lambda x: preprocessing(x, flag= FLAG))
    temp_df.columns = ['question1_cleaned_lemma', 'question1_cleaned_lemma_len']
    X_train = pd.concat([X_train, temp_df], axis=1)

    # text preprocessing for X_train que2
    temp_df = X_train.question2.apply(lambda x: preprocessing(x, flag= FLAG))
    temp_df.columns = ['question2_cleaned_lemma', 'question2_cleaned_lemma_len']
    X_train = pd.concat([X_train, temp_df], axis=1)

    # text preprocessing for X_test que1 
    # temp_df_t = X_test.question1.apply(lambda x: preprocessing(x, flag= FLAG))
    # temp_df_t.columns = ['question1_cleaned_lemma', 'question1_cleaned_lemma_len']
    # X_test = pd.concat([X_test, temp_df_t], axis=1)

    # text preprocessing for X_test que2
    # temp_df_t = X_test.question2.apply(lambda x: preprocessing(x, flag= FLAG))
    # temp_df_t.columns = ['question2_cleaned_stem', 'question2_cleaned_stem_len']
    # X_test = pd.concat([X_test, temp_df_t], axis=1)

    # Combining processed q1 and q2 
    X_train['q1_q2_combined'] = q1_q2_comb(X_train.question1_cleaned_lemma, X_train.question2_cleaned_lemma)
    # X_test['q1_q2_combined'] = q1_q2_comb(X_test.question1_cleaned_lemma, X_test.question2_cleaned_lemma)
    
    # Tokenising  combined q1 and q2
    X_train['q1_q2_combined_list'] = q1_q2_comb_token(X_train.q1_q2_combined)
    # X_test['q1_q2_combined_list'] = q1_q2_comb_token(X_test.q1_q2_combined)
    
    # creating word vectors
    word_vec = Word2Vec(X_train.q1_q2_combined_list, vector_size= VECTOR_SIZE, min_count=MIN_COUNT)

    # converting word vectors to document vectors
    X_train['doc_vector'] = X_train.q1_q2_combined_list.apply(lambda x: document_vector(x, word_vec.wv))
    # X_test['doc_vector'] = X_test.q1_q2_combined_list.apply(lambda x: document_vector(x, word_vec.wv))

    X_train_wv = list(X_train.doc_vector)
    # X_test_wv = list(X_test.doc_vector)

    lr = train_model(X_train_wv, y_train, estimator = LogisticRegression())


    print(lr.predict(X_train_wv))
    # print(X_test)
    # print(y_train)
    # print(y_test)


main(path = './data/train.csv')








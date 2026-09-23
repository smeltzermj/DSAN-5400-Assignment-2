import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=["author", "text"])
    df_test = pd.DataFrame(columns=["author", "text"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f) as fp:
                text = fp.read()
                # TODO If the directory name is either kennedy or johnson,
                #  create a pandas DataFrame where the column "authors" is
                #  the directory name and there is a field "text" which
                #  contains the text from the opened file. Note that you want
                #  a single DataFrame, but you loop over numerous files.
                if dir_name in ("kennedy", "johnson"):
                    df.loc[len(df)] = [dir_name, text]
                else:
                    # TODO Otherwise, we want to create a DataFrame for the
                    #  unlabeled data in a similar fashion. But this is a
                    #  little different because we don't get the label from
                    #  the directory but instead from the file name. Again,
                    #  the field "author" should have the author's name and
                    #  the field "text" should contain the text.
                    author = f.stem.split("_")[-1] 
                    df.loc[len(df)] = [author, text]
                    # In order to get the author name from the file, I had to take the name of the file, split it (with _ as the delimiter)
                    # And then take the last indexed value (i.e., the name). Tough!
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    # do the same for the test data
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))
    return df_train, df_test


def train_nb(df, alpha=0.1):
    """
    Takes as input a pandas DataFrame containing Federalist
    files text to determine priors and likelihoods
    :param df: a pandas DataFrame
    :return: two numpy arrays for the priors and likelihoods
    """
    # TODO Create a dictionary that maps whitespace-separated tokens in the
    #  file to a unique index. Also, create variables for the number of
    #  documents and the number of classes. Use df.shape for the vocabulary
    #  and the nunique() method for the number of classes
    vocabulary = {}
    word_index = 0

    for text in df["text"]:
        words = text.lower().split()

        for word in words:
            if word not in vocabulary:
                vocabulary[word] = word_index
                word_index += 1

    n_docs = df.shape[0]
    n_classes = df["author"].nunique()

    #print(len(vocabulary))
    #print(n_docs)
    #print(n_classes)


    # TODO Compute the priors
    priors = np.zeros(n_classes)

    for c in range(n_classes):
        priors[c] = df[df["author"] == c].shape[0] / n_docs
        

    # TODO Create a matrix containing all 0s called training_matrix of size
    #  (n_docs, len(vocabulary)), then fill it with the counts of each word
    #  for each document. This is the bag-of-words matrix for all the documents

    training_matrix = np.zeros((n_docs, len(vocabulary)))

    for doc_index, text in enumerate(df["text"]):
        words = text.lower().split()

        for word in words:
            word_index = vocabulary[word]
            training_matrix[doc_index, word_index] += 1

    #print(training_matrix.shape)
    #print(training_matrix[0].sum())
    
    # TODO Get word counts for both classes
    word_counts_per_class = np.zeros((n_classes, len(vocabulary)))

    for c in range(n_classes):
        word_counts_per_class[c] = training_matrix[df["author"] == c].sum(axis=0)

    #print(word_counts_per_class.shape)
    #print(word_counts_per_class[0].sum())
    #print(word_counts_per_class[1].sum())

    # TODO Initialize a matrix to store the likelihoods
    likelihoods = np.zeros((n_classes, len(vocabulary)))

    # TODO Then fill it in using Lidstone smoothing

    for c in range(n_classes):
        likelihoods[c] = (word_counts_per_class[c] + alpha) / (word_counts_per_class[c].sum() + alpha * len(vocabulary))

    #print(likelihoods.shape)
    #print(likelihoods[0].sum())
    #print(likelihoods[1].sum())


    return vocabulary, priors, likelihoods




def test(df, vocabulary, priors, likelihoods):
    """
    Takes as input a pandas DataFrame representing the disputed Federalist
    Papers and returns predictions for every text document
    :param df: a pandas DataFrame
    :return: a numpy array of predictions
    """
    class_predictions = []
    for text in df["text"]:
        test_vector = np.zeros(shape=(len(vocabulary)))
        # TODO Fill test_vector with counts for the words that appear in the
        #  vocabulary    
        words = text.lower().split() # tokenizing the speech

        for word in words:
            if word in vocabulary:
                word_index = vocabulary[word]
                test_vector[word_index] += 1

        # TODO Compute predictions p(y|text)
        preds = np.zeros(len(priors))

        for c in range(len(priors)):
            preds[c] = np.log(priors[c]) + np.sum(test_vector*np.log(likelihoods[c]))


        # TODO Then get your predictions, yhat
        yhat = np.argmax(preds)
        class_predictions.append(yhat)
    return class_predictions


def sklearn_nb(training_df, test_df):
    """
    Performs Naive Bayes classification using scikit-learn implementation
    :param training_df: training data
    :param test_df: test data
    :return: predictions
    """
    vectorizer = CountVectorizer()

    # TODO Fit the vectorizer on the training set text
    vectorizer.fit(training_df["text"])

    # TODO Then transform the text using the vectorizer
    training_data = vectorizer.transform(training_df["text"])
    training_data.toarray()

    # Do the same for the test data
    test_data = vectorizer.transform(test_df["text"])
    test_data.toarray()

    nb_classifier = MultinomialNB()
    # TODO Fit the Naive Bayes classifier
    nb_classifier.fit(training_data, training_df["author"])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb


def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    # TODO Compute performance measures
    accuracy = metrics.accuracy_score(true, preds)
    f1_score = metrics.f1_score(true, preds)
    conf_matrix = metrics.confusion_matrix(true, preds)

    return accuracy, f1_score, conf_matrix


def plot_confusion_matrix(conf_matrix_data, labels):
    """
    Takes as input confusion matrix data from get_metrics() and prints out a
    confusion matrix
    :param conf_matrix_data:
    :return: None
    """
    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data, annot=True)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.savefig("conf.jpg")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()

    training_df, test_df = build_dataframe(args.indir)
    vocabulary, priors, likelihoods = train_nb(training_df)
    class_predictions = test(test_df, vocabulary, priors, likelihoods)
    acc, f1, conf = get_metrics(test_df["author"], class_predictions)
    plot_confusion_matrix(conf, ["Kennedy", "Johnson"])
    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_acc, sklearn_f1, sklearn_conf = get_metrics(
        test_df["author"], sklearn_preds)




    # EDA for Problem 1C
    #def count_words(text):
    #    return len(text.split())
#
    #word_counts = []
#
    #for text in training_df["text"]:
    #    word_counts.append(count_words(text))
#
    #training_df["word_count"] = word_counts
#
    #print(training_df.head())
#
    #print(training_df.groupby("author")["word_count"].mean())
    #print(training_df.groupby("author")["word_count"].median())
#
    #kennedy_texts = training_df[training_df["author"] == 0]["text"]
    #kennedy_words = []
#
    #for text in kennedy_texts:
    #    words = text.lower().split()
    #    kennedy_words.extend(words)
#
    #kennedy_counts = Counter(kennedy_words)
    #print(kennedy_counts.most_common(20))
#
    #johnson_texts = training_df[training_df["author"] == 1]["text"]
    #johnson_words = []
#
    #for text in johnson_texts:
    #    words = text.lower().split()
    #    johnson_words.extend(words)
#
    #johnson_counts = Counter(johnson_words)
    #print(johnson_counts.most_common(20))
#
    #print(kennedy_counts["i"] / len(kennedy_words))
    #print(johnson_counts["i"] / len(johnson_words))
#
    #print(kennedy_counts["we"] / len(kennedy_words))
    #print(johnson_counts["we"] / len(johnson_words))
#
    #print(kennedy_counts["my"] / len(kennedy_words))
    #print(johnson_counts["my"] / len(johnson_words))
#
    #print(kennedy_counts["our"] / len(kennedy_words))
    #print(johnson_counts["our"] / len(johnson_words))
#
    #kennedy_first_words = kennedy_words[:15]
    #kennedy_last_words = kennedy_words[-15:]
    #johnson_first_words = johnson_words[:15]
    #johnson_last_words = johnson_words[-15:]

    #print(kennedy_first_words)
    #print(kennedy_last_words)
    #print(johnson_first_words)
    #print(johnson_last_words)
    # Turns out this wasn't the best method. Let's try iterating over a few speeches, instead.

    #for text in kennedy_texts.head(5):
    #    words = text.lower().split()
    #    print("Start: ", words[:15])
    #    print("End: ", words[-15:])
#
    #for text in johnson_texts.head(5):
    #    words = text.lower().split()
    #    print("Start: ", words[:15])
    #    print("End: ", words[-15:])



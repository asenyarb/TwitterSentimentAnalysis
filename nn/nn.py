import random
import time
from operator import itemgetter
from itertools import combinations
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Concatenate
import re
from functools import reduce
import string
# DataFrame
import tensorflow.keras.models
import pandas as pd

# Matplot
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import csv
import json
import subprocess
import re
import numpy as np
from collections import Counter
import time
import pickle
import itertools


class NNEnglish:
    # DATASET
    DATASET_PATH = './data/data_en.csv'
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    TRAIN_SIZE = 0.8

    # TEXT CLEANING
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    # WORD2VEC
    W2V_SIZE = 300
    W2V_WINDOW = 7
    W2V_EPOCH = 32
    W2V_MIN_COUNT = 10

    # KERAS
    SEQUENCE_LENGTH = 300
    EPOCHS = 8
    BATCH_SIZE = 120

    # SENTIMENT
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    SENTIMENT_THRESHOLDS = (0.4, 0.7)

    # EXPORT
    KERAS_MODEL = "./nn_models/en/model.h5"
    KERAS_CONV_MODEL = "./nn_models/en/conv_model.h5"
    KERAS_REC_CONV_MODEL = "./nn_models/en/rec_conv_model.h5"
    WORD2VEC_MODEL = "./nn_models/en/model.w2v"
    TOKENIZER_MODEL = "./nn_models/en/tokenizer.pkl"
    ENCODER_MODEL = "./nn_models/en/encoder.pkl"

    DECODE_MAP = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
    stop_words = None
    stemmer = SnowballStemmer("english")
    w2v_model: gensim.models.word2vec.Word2Vec = None
    df_train: pd.DataFrame = None
    df_test: pd.DataFrame = None
    tokenizer: Tokenizer = None
    vocab_size: int = None
    encoder: LabelEncoder = None
    recurrent_model: Sequential = None
    convolutional_model: Sequential = None
    convolutional_recurrent_model: Sequential = None

    @classmethod
    def initialize(cls):
        try:
            cls.stop_words = stopwords.words("english")
        except LookupError:
            nltk.download('stopwords')
            cls.stop_words = stopwords.words("english")
        cls.__setup_models()

    @classmethod
    def predict(cls, model, text, include_neutral=True):
        start_at = time.time()
        # Tokenize text
        x_test = pad_sequences(cls.tokenizer.texts_to_sequences([text]), maxlen=cls.SEQUENCE_LENGTH)
        # Predict
        score = model.predict([x_test])[0]
        # Decode sentiment
        label = cls.__score_to_label(score, include_neutral=include_neutral)

        return {"label": label, "score": float(score),
                "elapsed_time": time.time() - start_at}

    @classmethod
    def __load_train_data(cls):
        print("Open file:", cls.DATASET_PATH)
        df = pd.read_csv(cls.DATASET_PATH, encoding=cls.DATASET_ENCODING, names=cls.DATASET_COLUMNS)
        print("Dataset size:", len(df))

        df.target = df.target.apply(lambda x: cls.__decode_sentiment(x))

        # target_cnt = Counter(df.target)
        # plt.figure(figsize=(16, 8))
        # plt.bar(target_cnt.keys(), target_cnt.values())
        # plt.title("Dataset labels distribuition")

        df.text = df.text.apply(lambda x: cls.__preprocess_text(x))

        cls.df_train, cls.df_test = train_test_split(df, test_size=1 - cls.TRAIN_SIZE, random_state=42)
        print("TRAIN size:", len(cls.df_train))
        print("TEST size:", len(cls.df_test))

    @classmethod
    def __setup_w2v_model(cls):
        try:
            print("Loading exising W2V model")
            cls.w2v_model = gensim.models.word2vec.Word2Vec.load(cls.WORD2VEC_MODEL)
        except FileNotFoundError:
            print("W2V model not found. Creating a new one")

            if cls.df_train is None:
                print("Train data not found. Loading.")
                cls.__load_train_data()
                print('Train data loaded')
            documents = [_text.split() for _text in cls.df_train.text]

            w2v_model = gensim.models.word2vec.Word2Vec(
                vector_size=cls.W2V_SIZE, window=cls.W2V_WINDOW, min_count=cls.W2V_MIN_COUNT, workers=8
            )
            w2v_model.build_vocab(documents)

            print("Vocab size", len(w2v_model.wv.key_to_index))

            w2v_model.train(documents, total_examples=len(documents), epochs=cls.W2V_EPOCH)
            w2v_model.save(cls.WORD2VEC_MODEL)

            cls.w2v_model = w2v_model

        print("W2V model loaded")

    @classmethod
    def __setup_tokenizer(cls):
        try:
            print("Loading existing Tokenizer")
            cls.tokenizer = pickle.load(open(cls.TOKENIZER_MODEL, 'rb'))
        except FileNotFoundError:
            print("Tokenizer not found. Creating a new one")

            if cls.df_train is None:
                print("Train data not found. Loading.")
                cls.__load_train_data()
                print('Train data loaded')

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(cls.df_train.text)

            pickle.dump(tokenizer, open(cls.TOKENIZER_MODEL, "wb"), protocol=0)

            cls.tokenizer = tokenizer

        cls.vocab_size = len(cls.tokenizer.word_index) + 1
        print("Total words", cls.vocab_size)
        print("Tokenizer loaded")

    @classmethod
    def __setup_label_encoder(cls):
        try:
            print("Loading existing LabelEncoder")
            cls.encoder = pickle.load(open(cls.ENCODER_MODEL, 'rb'))
        except FileNotFoundError:
            print("LabelEncoder not found. Creating a new one")

            if cls.df_train is None:
                print("Train data not found. Loading.")
                cls.__load_train_data()
                print('Train data loaded')

            encoder = LabelEncoder()
            encoder.fit(cls.df_train.target.tolist())

            pickle.dump(encoder, open(cls.ENCODER_MODEL, "wb"), protocol=0)

            cls.encoder = encoder
        print("LabelEncoder loaded")

    @classmethod
    def __setup_models(cls):
        cls.__setup_tokenizer()
        try:
            print('Loading rnn model')
            cls.recurrent_model = tensorflow.keras.models.load_model(cls.KERAS_MODEL)
        except (IOError, ImportError):
            print('Unable to load rnn model. Creating and training a new one')
            cls.__create_and_train_recurrent_model()
        try:
            print('Loading cnn model')
            cls.convolutional_model = tensorflow.keras.models.load_model(cls.KERAS_CONV_MODEL)
        except (IOError, ImportError):
            print('Unable to load cnn model. Creating and training a new one')
            cls.__create_and_train_convolutional_model()
        try:
            print('Loading rnn + cnn model')
            cls.convolutional_recurrent_model = tensorflow.keras.models.load_model(cls.KERAS_REC_CONV_MODEL)
        except (IOError, ImportError):
            print('Unable to load rnn+cnn model. Creating and training a new one')
            cls.__create_and_train_convolutional_recurrent_model()

    @classmethod
    def __create_and_train_recurrent_model(cls):
        cls.__load_train_data()
        cls.__setup_w2v_model()
        cls.__setup_tokenizer()
        cls.__setup_label_encoder()

        assert cls.df_train is not None
        assert cls.w2v_model is not None
        assert cls.tokenizer is not None
        assert cls.vocab_size
        assert cls.encoder is not None

        x_train = pad_sequences(cls.tokenizer.texts_to_sequences(cls.df_train.text), maxlen=cls.SEQUENCE_LENGTH)
        x_test = pad_sequences(cls.tokenizer.texts_to_sequences(cls.df_test.text), maxlen=cls.SEQUENCE_LENGTH)
        labels = cls.df_train.target.unique().tolist()
        labels.append(cls.NEUTRAL)

        y_train = cls.encoder.transform(cls.df_train.target.tolist())
        y_test = cls.encoder.transform(cls.df_test.target.tolist())

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        print("x_train", x_train.shape)
        print("y_train", y_train.shape, '\n')
        print("x_test", x_test.shape)
        print("y_test", y_test.shape)

        embedding_layer = cls.__get_embedding_layer()
        model = Sequential()
        model.add(embedding_layer)
        model.add(Dropout(0.5))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)
        ]
        history = model.fit(x_train, y_train,
                            batch_size=cls.BATCH_SIZE,
                            epochs=cls.EPOCHS,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=callbacks)

        model.save(cls.KERAS_MODEL)

        score = model.evaluate(x_test, y_test, batch_size=cls.BATCH_SIZE)
        print("ACCURACY:", score[1])
        print("LOSS:", score[0])

        cls.recurrent_model = model
        cls.__plot_learning_graphs(model, history, x_test)

    @classmethod
    def __create_and_train_convolutional_model(cls):
        cls.__load_train_data()
        cls.__setup_w2v_model()
        cls.__setup_tokenizer()
        cls.__setup_label_encoder()

        assert cls.df_train is not None
        assert cls.w2v_model is not None
        assert cls.tokenizer is not None
        assert cls.vocab_size
        assert cls.encoder is not None

        x_train = pad_sequences(cls.tokenizer.texts_to_sequences(cls.df_train.text), maxlen=cls.SEQUENCE_LENGTH)
        x_test = pad_sequences(cls.tokenizer.texts_to_sequences(cls.df_test.text), maxlen=cls.SEQUENCE_LENGTH)
        labels = cls.df_train.target.unique().tolist()
        labels.append(cls.NEUTRAL)

        y_train = cls.encoder.transform(cls.df_train.target.tolist())
        y_test = cls.encoder.transform(cls.df_test.target.tolist())

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        print("x_train", x_train.shape)
        print("y_train", y_train.shape, '\n')
        print("x_test", x_test.shape)
        print("y_test", y_test.shape)

        embedding_layer = cls.__get_embedding_layer()
        model = Sequential()
        model.add(embedding_layer)
        model.add(Dropout(0.5))
        model.add(Conv1D(100, 2, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)
        ]
        history = model.fit(x_train, y_train,
                            batch_size=cls.BATCH_SIZE,
                            epochs=cls.EPOCHS,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=callbacks)

        model.save(cls.KERAS_CONV_MODEL)

        score = model.evaluate(x_test, y_test, batch_size=cls.BATCH_SIZE)
        print("ACCURACY:", score[1])
        print("LOSS:", score[0])

        cls.convolutional = model
        cls.__plot_learning_graphs(model, history, x_test)

    @classmethod
    def __create_and_train_convolutional_recurrent_model(cls):
        cls.__load_train_data()
        cls.__setup_w2v_model()
        cls.__setup_tokenizer()
        cls.__setup_label_encoder()

        assert cls.df_train is not None
        assert cls.w2v_model is not None
        assert cls.tokenizer is not None
        assert cls.vocab_size
        assert cls.encoder is not None

        x_train = pad_sequences(cls.tokenizer.texts_to_sequences(cls.df_train.text), maxlen=cls.SEQUENCE_LENGTH)
        x_test = pad_sequences(cls.tokenizer.texts_to_sequences(cls.df_test.text), maxlen=cls.SEQUENCE_LENGTH)
        labels = cls.df_train.target.unique().tolist()
        labels.append(cls.NEUTRAL)

        y_train = cls.encoder.transform(cls.df_train.target.tolist())
        y_test = cls.encoder.transform(cls.df_test.target.tolist())

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        print("x_train", x_train.shape)
        print("y_train", y_train.shape, '\n')
        print("x_test", x_test.shape)
        print("y_test", y_test.shape)

        embedding_layer = cls.__get_embedding_layer()

        #####

        input_layer = Input(x_train.shape[1:])

        ######
        tower1 = embedding_layer(input_layer)
        tower1 = Dropout(0.5)(tower1)
        tower1 = Conv1D(
            100, 2, padding='valid', activation='relu', strides=1
        )(tower1)
        tower1 = GlobalMaxPooling1D()(tower1)
        tower1 = Dense(24, activation='relu')(tower1)

        ######
        tower2 = embedding_layer(input_layer)
        tower1 = Dropout(0.5)(tower1)
        tower2 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(tower2)

        ######
        merged = Concatenate()([tower1, tower2])
        merged = Dense(5, activation='relu')(merged)
        out = Dense(1, activation='sigmoid')(merged)

        model = Model(input_layer, out)

        model.compile(
            loss='binary_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)
        ]
        history = model.fit(x_train, y_train,
                            batch_size=int(cls.BATCH_SIZE / 2),
                            epochs=cls.EPOCHS,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=callbacks)

        model.save(cls.KERAS_REC_CONV_MODEL)

        score = model.evaluate(x_test, y_test, batch_size=cls.BATCH_SIZE)
        print("ACCURACY:", score[1])
        print("LOSS:", score[0])

        cls.convolutional_recurrent_model = model
        cls.__plot_learning_graphs(model, history, x_test)

    @classmethod
    def __get_embedding_layer(cls):
        embedding_matrix = np.zeros((cls.vocab_size, cls.W2V_SIZE))
        for word, i in cls.tokenizer.word_index.items():
            if word in cls.w2v_model.wv:
                embedding_matrix[i] = cls.w2v_model.wv[word]
        return Embedding(
            cls.vocab_size, cls.W2V_SIZE, weights=[embedding_matrix],
            input_length=cls.SEQUENCE_LENGTH, trainable=False
        )

    @classmethod
    def __score_to_label(cls, score, include_neutral=True):
        if include_neutral:
            label = cls.NEUTRAL
            if score <= cls.SENTIMENT_THRESHOLDS[0]:
                label = cls.NEGATIVE
            elif score >= cls.SENTIMENT_THRESHOLDS[1]:
                label = cls.POSITIVE

            return label
        else:
            return cls.NEGATIVE if score < 0.5 else cls.POSITIVE

    @classmethod
    def __decode_sentiment(cls, label):
        return cls.DECODE_MAP[int(label)]

    @classmethod
    def __preprocess_text(cls, text, stem=False):
        # Remove link, user and special characters
        text = re.sub(cls.TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in cls.stop_words:
                if stem:
                    tokens.append(cls.stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    @classmethod
    def __plot_learning_graphs(cls, model, history, x_test):
        cls.__save_learning_results_plot(
            acc=history.history['accuracy'],
            val_acc=history.history['val_accuracy'],
            loss=history.history['loss'],
            val_loss=history.history['val_loss']
        )

        y_test_1d = list(cls.df_test.target)
        scores = model.predict(x_test, verbose=1, batch_size=8000)
        y_pred_1d = [cls.__score_to_label(score, include_neutral=False) for score in scores]

        cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
        cls.__save_confusion_matrix_plot(cnf_matrix, classes=cls.df_train.target.unique(), title="Confusion matrix")

        print(classification_report(y_test_1d, y_pred_1d))
        print(accuracy_score(y_test_1d, y_pred_1d))

    @staticmethod
    def __save_learning_results_plot(acc, val_acc, loss, val_loss):
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig('./imgs/training_graph.png')

    @classmethod
    def __save_confusion_matrix_plot(
        cls, cm, classes, title='Confusion matrix', cmap=plt.get_cmap('Blues')
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=30)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
        plt.yticks(tick_marks, classes, fontsize=22)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Predicted label', fontsize=25)
        plt.savefig('./imgs/training_graph_squares.png')


class NNRussian:
    russian_alphabet = 'абвгдефжзийклмнопрстуфхцчшщъыьэюя'
    symbols_values = {
        letter: index for index, letter in enumerate(
            russian_alphabet + string.ascii_letters + string.punctuation + string.digits,
            1
        )
    }
    max_symbol_value = max(symbols_values.values())
    multiple_dots_or_commas_pattern = re.compile('([.,=:;(){}P]+|@[A-z]+)')
    matrix_rows_n = 20
    matrix_cols_n = 20

    @staticmethod
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    @classmethod
    def preprocess_sentence(cls, sentence):
        return re.sub(cls.multiple_dots_or_commas_pattern, '', sentence.replace(' - ', ' ')).strip()

    @staticmethod
    def preprocess_words(words_list, max_word_length):
        return [
            w for word in words_list
            if (w := word.strip()[:max_word_length])
        ]

    @classmethod
    def preprocess_sentences(cls, sentences, max_words, max_word_length):
        for sentence in sentences:
            sentence = cls.preprocess_sentence(sentence)
            if not sentence:
                continue
            sentence = sentence.split(' ')
            sentence = cls.preprocess_words(sentence, max_word_length)
            if sentence and len(sentence) <= max_words:
                yield sentence

    @classmethod
    def process_word_to_numbers_list(cls, word, max_length):
        assert len(word) <= max_length
        return [
            cls.symbols_values.get(symbol, 0) / cls.max_symbol_value for symbol in word
        ] + [0] * (max_length - len(word))

    @classmethod
    def process_words_to_matrix(cls, words, matrix_rows_n: int, matrix_cols_n: int):
        # if there's more words then nn is able to process (words number > matrix rows number))
        # array of combinations of words is returned, else array of 1 item is returned: [<matrix for sentence>]
        assert len(words) and len(words) <= matrix_rows_n

        words = words + [''] * max(matrix_rows_n - len(words), 0)

        return [cls.process_word_to_numbers_list(w, matrix_cols_n) for w in words]

    @classmethod
    def process_single_sentence(cls, sentence):
        sentence = cls.preprocess_sentence(sentence)
        if not sentence:
            return None
        sentence = sentence.split(' ')
        sentence = cls.preprocess_words(sentence, cls.matrix_cols_n)[:cls.matrix_rows_n]
        return cls.process_words_to_matrix(sentence, cls.matrix_rows_n, cls.matrix_cols_n)

    @classmethod
    def prepare_test_data(cls, positives_list_filepath, others_list_filepath, matrix_rows_n: int, matrix_cols_n: int):
        # returns 2 arrays. first is dataset to process, second - expected outputs
        with open(positives_list_filepath, 'r') as positives_f, open(others_list_filepath, 'r') as others_f:
            positives_list = list(positives_f)
            others_list = list(others_f)

        positives_list = list(cls.preprocess_sentences(positives_list, matrix_rows_n, matrix_cols_n))
        others_list = list(cls.preprocess_sentences(others_list, matrix_rows_n, matrix_cols_n))

        assert len(positives_list) and len(others_list)

        data_in = [
            cls.process_words_to_matrix(words, matrix_rows_n, matrix_cols_n)
            for words in positives_list + others_list
        ]

        arr_1 = data_in

        arr_2 = [1.] * len(positives_list) + [0.] * len(others_list)

        a = list(zip(arr_1, arr_2))
        random.shuffle(a)
        return list(map(itemgetter(0), a)), list(map(itemgetter(1), a))

    @staticmethod
    def teach_cnn_model_and_save(test_results, matrix_rows_n, matrix_cols_n, model_name):
        normalized_data_to_process, comparison_results = test_results
        x = np.array(normalized_data_to_process)
        y = np.array(comparison_results)

        model = Sequential()

        model.add(Conv1D(100, 2, padding='valid', input_shape=(matrix_rows_n, matrix_cols_n), activation='relu', strides=1))

        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 100
        model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.10, verbose=2)
        model.save(model_name)

    @staticmethod
    def teach_recursive_model_and_save(test_results, matrix_rows_n, matrix_cols_n, model_name):
        normalized_data_to_process, comparison_results = test_results
        x = np.array(normalized_data_to_process)
        y = np.array(comparison_results)

        model = Sequential()
        model.add(LSTM(128))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 100
        model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.10, verbose=2)
        model.save(model_name)

    @staticmethod
    def teach_conv_recursive_model_and_save(test_results, matrix_rows_n, matrix_cols_n, model_name):
        normalized_data_to_process, comparison_results = test_results
        x = np.array(normalized_data_to_process)
        y = np.array(comparison_results)

        input_shape = Input(shape=(matrix_rows_n, matrix_cols_n))

        tower1 = Conv1D(
            100, 2, padding='valid', input_shape=(matrix_rows_n, matrix_cols_n), activation='relu', strides=1
        )(input_shape)
        tower1 = GlobalMaxPooling1D()(tower1)
        tower1 = Dense(256, activation='relu')(tower1)

        tower2 = LSTM(128)(input_shape)
        tower2 = Dense(256, activation='relu')(tower2)

        merged = Concatenate()([tower1, tower2])

        out = Dense(256, activation='relu')(merged)
        out = Dense(1, activation='sigmoid')(out)

        model = Model(input_shape, out)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 100
        model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.10, verbose=2)
        model.save(model_name)

    @classmethod
    def prepare_data_and_teach_model(
        cls,
        path_to_teach_for, neutral_tweets_path,
        model_name
    ):
        matrix_rows_number, matrix_cols_number = cls.matrix_rows_n, cls.matrix_cols_n
        d = cls.prepare_test_data(
            path_to_teach_for, neutral_tweets_path,
            matrix_rows_number, matrix_cols_number
        )

        print('prepared data')

        cls.teach_cnn_model_and_save(d, matrix_rows_number, matrix_cols_number, f'{model_name}_cnn')
        cls.teach_recursive_model_and_save(d, matrix_rows_number, matrix_cols_number, f'{model_name}_rnn')
        cls.teach_conv_recursive_model_and_save(d, matrix_rows_number, matrix_cols_number, f'{model_name}_cnn_rnn')

        print(f'saved model to {model_name}/')


NNEnglish.initialize()

import itertools

# Matplot
import matplotlib.pyplot as plt

# DataFrame
import tensorflow.keras.models
import pandas as pd
from multiprocessing import Pool


from nltk.wsd import lesk
from nltk.tokenize import word_tokenize, sent_tokenize

#import spacy

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, Conv2D, MaxPooling1D, LSTM
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.vis_utils import plot_model

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2vec
import gensim
import gensim.downloader

# Utility
import re
import time
import pickle
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def preprocess_text(_df):
    _df['text'] = _df['text'].apply(NNEnglish.preprocess_text)
    return _df


class NNEnglish:
    # DATASET
    DATASET_PATH = './data/data_en.csv'
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    TRAIN_SIZE = 0.8

    # TEXT CLEANING
    TEXT_CLEANING_RE = re.compile(r"@\S+|https?:\S+|http?:\S")  #"|[^A-Za-z0-9]+")
    TEXT_CLEANING_RE2 = re.compile(r'[^a-zA-Z0-9 ]+')

    # WORD2VEC
    W2V_SIZE = 300
    W2V_WINDOW = 7
    W2V_EPOCH = 32
    W2V_MIN_COUNT = 10

    # KERAS
    SEQUENCE_LENGTH = 300
    EPOCHS = 5
    BATCH_SIZE = 300
    BATCH_SIZE_REC = 100

    # SENTIMENT
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    SENTIMENT_THRESHOLDS = (0.4, 0.7)

    # EXPORT
    DF_PREPROCESSED_SAVE_PATH = './data/preprocessed_df_pickle.pkl'
    KERAS_CONV_MODEL = "./nn_models/en/conv_model.h5"
    KERAS_REC_MODEL = "./nn_models/en/rec_model.h5"
    KERAS_REC_CONV_MODEL = "./nn_models/en/rec_conv_model.h5"
    WORD2VEC_MODEL = "./nn_models/en/model.w2v"
    TOKENIZER_MODEL = "./nn_models/en/tokenizer.pkl"
    ENCODER_MODEL = "./nn_models/en/encoder.pkl"

    DECODE_MAP = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
    stop_words = None
    stemmer = SnowballStemmer("english")
    w2v = None
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
    def __setup_lesk(cls):
        try:
            lesk(word_tokenize('This device is used to jam the signal'), 'jam')
        except LookupError:
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            lesk(word_tokenize('This device is used to jam the signal'), 'jam')

    @classmethod
    def __remove_repetitive_chars_if_unknown(cls, word, sentence_tokenized):
        if lesk(sentence_tokenized, word) is not None:
            return word

        s = ''
        prev_char = None
        for c in word:
            if prev_char != c:
                s += c
            prev_char = c
        return s

    @classmethod
    def preprocess_text(cls, text, stem=False):
        # Remove link, user and special characters
        text = re.sub(cls.TEXT_CLEANING_RE, ' ', str(text).lower())
        text = re.sub(cls.TEXT_CLEANING_RE2, '', text).strip()

        words = []
        total_words_in_text, words_not_in_w2v = 0, 0
        for sentence in sent_tokenize(text, 'english'):
            sentence_tokenized = filter(
                None, map(
                    lambda w: w.strip(), word_tokenize(sentence, 'english')
                )
            )
            for word in filter(lambda x: x not in cls.stop_words, sentence_tokenized):
                total_words_in_text += 1
                if word not in cls.w2v.key_to_index:
                    words_not_in_w2v += 1
                #word = cls.__remove_repetitive_chars_if_unknown(word, sentence_tokenized)
                #if stem:
                #    word = cls.stemmer.stem(word)
                words.append(word)

        if not total_words_in_text or words_not_in_w2v / total_words_in_text > 0.5:
            return ""
        return " ".join(words)

    @classmethod
    def __setup_w2v_model(cls):
        print('Loading W2V')
        cls.w2v = gensim.downloader.load('word2vec-google-news-300')

        print("W2V model loaded")

    @classmethod
    def __load_train_data(cls):
        if cls.df_train is not None and cls.df_test is not None:
            print('TRAIN DATA ALREADY LOADED. SKIPPING')
            print("TRAIN size:", len(cls.df_train))
            print("TEST size:", len(cls.df_test))
            return

        try:
            print('Loading preprocessed saved df...')
            with open(cls.DF_PREPROCESSED_SAVE_PATH, 'rb') as df_pickle_train:
                df = pickle.load(df_pickle_train)
        except Exception:
            print('Unable to load preprocessed saved df. Creating a new one')
            print("Open file:", cls.DATASET_PATH)
            df = pd.read_csv(cls.DATASET_PATH, encoding=cls.DATASET_ENCODING, names=cls.DATASET_COLUMNS)
            df['target'] = df['target'].apply(lambda x: cls.__decode_sentiment(x))
            # target_cnt = Counter(df.target)
            # plt.figure(figsize=(16, 8))
            # plt.bar(target_cnt.keys(), target_cnt.values())
            # plt.title("Dataset labels distribuition")
            print(df['text'])

            df = parallelize_dataframe(df, preprocess_text)
            #df.text = df.text.apply(lambda x: cls.__preprocess_text(x))
            print(df['text'])
            df['text'].replace('', np.nan, inplace=True)
            df.dropna(subset=['text'], inplace=True)

            with open(cls.DF_PREPROCESSED_SAVE_PATH, 'wb') as df_pickle_train:
                pickle.dump(df, df_pickle_train)

        print("Dataset size:", len(df))
        cls.df_train, cls.df_test = train_test_split(df, test_size=1 - cls.TRAIN_SIZE, random_state=42)
        print("TRAIN size:", len(cls.df_train))
        print("TEST size:", len(cls.df_test))

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
        cls.__setup_w2v_model()
        cls.__load_train_data()
        cls.__setup_tokenizer()
        cls.__setup_label_encoder()
        try:
            print('Loading cnn model...')
            raise ImportError('as')
            cls.convolutional_model = tensorflow.keras.models.load_model(cls.KERAS_CONV_MODEL)
        except (IOError, ImportError):
            print('Unable to load cnn model. Creating and training a new one')
            cls.__create_and_train_convolutional_model()
        #try:
        #    print('Loading rnn model...')
        #    raise ImportError('as')
        #    cls.recurrent_model = tensorflow.keras.models.load_model(cls.KERAS_REC_MODEL)
        #except (IOError, ImportError):
        #    print('Unable to load rnn model. Creating and training a new one')
        #    cls.__create_and_train_recurrent_model()

    @classmethod
    def __get_embedding_layer(cls):
        print('Getting embedding layer...')
        embedding_matrix = np.zeros((cls.vocab_size, cls.w2v.vector_size))

        words_count, unknown_words_count = 0, 0
        for word, i in cls.tokenizer.word_index.items():
            words_count += 1
            if word in cls.w2v.key_to_index:
                embedding_matrix[i] = cls.w2v.get_vector(word)
            else:
                unknown_words_count += 1

        print(f'Words total: {words_count}, where unknown = {unknown_words_count}')

        layer = Embedding(
            cls.vocab_size, cls.w2v.vector_size, weights=[embedding_matrix],
            input_length=cls.SEQUENCE_LENGTH, trainable=False
        )
        print('Built embedding layer')
        return layer

    @classmethod
    def __create_and_train_convolutional_model(cls):
        assert cls.df_train is not None
        assert cls.w2v is not None
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

        model = Sequential()
        model.add(cls.__get_embedding_layer())
        #model.add(Dropout(0.5))
        model.add(Conv1D(50, 100, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        print(f"Here's the model summary:\n{model.summary()}")
        plot_model(model, to_file='conv_model_plot.png', show_shapes=True, show_layer_names=True)

        model.compile(
            loss='binary_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)
        ]
        history = model.fit(x_train, y_train,
                            batch_size=cls.BATCH_SIZE,
                            epochs=cls.EPOCHS,
                            validation_split=0.1, verbose=1, callbacks=callbacks)

        print(
            history.history['accuracy'],
            history.history['val_accuracy'],
            history.history['loss'],
            history.history['val_loss']
        )

        model.save(cls.KERAS_CONV_MODEL)

        score = model.evaluate(x_test, y_test, batch_size=cls.BATCH_SIZE)
        print("ACCURACY:", score[1])
        print("LOSS:", score[0])

        cls.convolutional_model = model
        cls.__plot_learning_graphs('conv', model, history, x_test)

    @classmethod
    def __create_and_train_recurrent_model(cls):
        assert cls.df_train is not None
        assert cls.w2v is not None
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

        model = Sequential()
        model.add(cls.__get_embedding_layer())
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))

        print(f"Here's the model summary:\n{model.summary()}")

        plot_model(model, to_file='rec_model_plot.png', show_shapes=True, show_layer_names=True)

        model.compile(
            loss='binary_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)
        ]
        history = model.fit(x_train, y_train,
                            batch_size=cls.BATCH_SIZE_REC,
                            epochs=cls.EPOCHS,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=callbacks)

        print(
            history.history['accuracy'],
            history.history['val_accuracy'],
            history.history['loss'],
            history.history['val_loss']
        )

        model.save(cls.KERAS_REC_MODEL)

        score = model.evaluate(x_test, y_test, batch_size=cls.BATCH_SIZE_REC)
        print("ACCURACY:", score[1])
        print("LOSS:", score[0])

        scores = model.predict(x_test, verbose=1, batch_size=cls.BATCH_SIZE)

        cls.recurrent_model = model
        cls.__plot_learning_graphs('rec', model, history, x_test)

        #print(cls.df_train)

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
    def __plot_learning_graphs(cls, model_name, model, history, x_test):
        cls.__save_learning_results_plot(
            model_name,
            acc=history.history['accuracy'],
            val_acc=history.history['val_accuracy'],
            loss=history.history['loss'],
            val_loss=history.history['val_loss']
        )

        y_test_1d = list(cls.df_test.target)
        scores = model.predict(x_test, verbose=1, batch_size=cls.BATCH_SIZE)
        y_pred_1d = [cls.__score_to_label(score, include_neutral=False) for score in scores]

        cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
        cls.__save_confusion_matrix_plot(
            model_name,
            cnf_matrix,
            classes=cls.df_train.target.unique(),
            title="Confusion matrix"
        )

        print(classification_report(y_test_1d, y_pred_1d))
        print(accuracy_score(y_test_1d, y_pred_1d))

    @staticmethod
    def __save_learning_results_plot(model_name, acc, val_acc, loss, val_loss):
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

        plt.savefig(f'./imgs/training_graph_{model_name}.png')

    @classmethod
    def __save_confusion_matrix_plot(
            cls, model_name, cm, classes, title='Confusion matrix', cmap=plt.get_cmap('Blues')
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
        plt.savefig(f'./imgs/training_graph_squares_{model_name}.png')


NNEnglish.initialize()
#if __name__ == '__main__':
#    res = NNEnglish.predict(NNEnglish.convolutional_model, "Man, I like this movie", include_neutral=False)
#    print(res)

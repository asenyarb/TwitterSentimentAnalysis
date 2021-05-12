import random
import time
from operator import itemgetter
from itertools import combinations
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Concatenate
import numpy as np
import re
import string

russian_alphabet = 'абвгдефжзийклмнопрстуфхцчшщъыьэюя'
symbols_values = {
    letter: index for index, letter in enumerate(
        russian_alphabet + string.ascii_letters + string.punctuation + string.digits,
        1
    )
}
max_symbol_value = max(symbols_values.values())
multiple_dots_or_commas_pattern = re.compile('([.,=:;(){}P]+|@[A-z]+)')


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def preprocess_sentence(sentence):
    return re.sub(multiple_dots_or_commas_pattern, '', sentence.replace(' - ', ' ')).strip()


def preprocess_words(words_list, max_word_length):
    return [
        w for word in words_list
        if (w := word.strip()[:max_word_length])
    ]


def preprocess_sentences(sentences, max_words, max_word_length):
    for sentence in sentences:
        sentence = preprocess_sentence(sentence)
        if not sentence:
            continue
        sentence = sentence.split(' ')
        sentence = preprocess_words(sentence, max_word_length)
        if sentence and len(sentence) <= max_words:
            yield sentence


def process_word_to_numbers_list(word, max_length):
    assert len(word) <= max_length
    return [
        symbols_values.get(symbol, 0) / max_symbol_value for symbol in word
    ] + [0] * (max_length - len(word))


def process_words_to_matrix(words, matrix_rows_n: int, matrix_cols_n: int):
    # if there's more words then nn is able to process (words number > matrix rows number))
    # array of combinations of words is returned, else array of 1 item is returned: [<matrix for sentence>]
    assert len(words) and len(words) <= matrix_rows_n

    words = words + [''] * max(matrix_rows_n - len(words), 0)

    return [process_word_to_numbers_list(w, matrix_cols_n) for w in words]


def process_single_sentence(sentence):
    sentence = preprocess_sentence(sentence)
    if not sentence:
        return None
    sentence = sentence.split(' ')
    sentence = preprocess_words(sentence, matrix_cols_n)[:matrix_rows_n]
    return process_words_to_matrix(sentence, matrix_rows_n, matrix_cols_n)


def prepare_test_data(positives_list_filepath, others_list_filepath, matrix_rows_n: int, matrix_cols_n: int):
    # returns 2 arrays. first is dataset to process, second - expected outputs
    with open(positives_list_filepath, 'r') as positives_f, open(others_list_filepath, 'r') as others_f:
        positives_list = list(positives_f)
        others_list = list(others_f)

    positives_list = list(preprocess_sentences(positives_list, matrix_rows_n, matrix_cols_n))
    others_list = list(preprocess_sentences(others_list, matrix_rows_n, matrix_cols_n))

    assert len(positives_list) and len(others_list)

    data_in = [
        process_words_to_matrix(words, matrix_rows_n, matrix_cols_n)
        for words in positives_list + others_list
    ]

    arr_1 = data_in

    arr_2 = [1.] * len(positives_list) + [0.] * len(others_list)

    a = list(zip(arr_1, arr_2))
    random.shuffle(a)
    return list(map(itemgetter(0), a)), list(map(itemgetter(1), a))


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


def prepare_data_and_teach_model(
    path_to_teach_for, neutral_tweets_path,
    matrix_rows_number, matrix_cols_number,
    model_name
):
    d = prepare_test_data(
        path_to_teach_for, neutral_tweets_path,
        matrix_rows_number, matrix_cols_number
    )

    print('prepared data')

    teach_cnn_model_and_save(d, matrix_rows_number, matrix_cols_number, f'{model_name}_cnn')
    teach_recursive_model_and_save(d, matrix_rows_number, matrix_cols_number, f'{model_name}_rnn')
    teach_conv_recursive_model_and_save(d, matrix_rows_number, matrix_cols_number, f'{model_name}_cnn_rnn')

    print(f'saved model to {model_name}/')


matrix_rows_n = 20
matrix_cols_n = 20

if __name__ == '__main__':
    positives_path, neutral_path, negatives_path = (
        './data/positive_tweets_list.txt',
        './data/neutral_tweets_list.txt',
        './data/negative_tweets_list.txt'
    )

    prepare_data_and_teach_model(
        positives_path, neutral_path,
        matrix_rows_n, matrix_cols_n,
        './nn/positives_nn'
    )

    prepare_data_and_teach_model(
        negatives_path, neutral_path,
        matrix_rows_n, matrix_cols_n,
        './nn/negatives_nn'
    )

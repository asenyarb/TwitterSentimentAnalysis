from tensorflow.keras.models import load_model

from csv_parse.csv_parse import CSVParseRU
from nn.nn import NNRussian, NNEnglish

nn_types = {
    'cnn': 'Свёрточная',
    'rnn': 'Рекуррентная',
    'cnn_rnn': 'Свёрточная и Рекуррентная параллельно'
}


def analyze_sentence_positive(processed_sentence, nn_type):
    model = load_model(f'nn/positives_nn_{nn_type}')
    return model.predict([processed_sentence])[0][0]


def analyze_sentence_negative(processed_sentence, nn_type):
    model = load_model(f'nn/negatives_nn_{nn_type}')
    return model.predict([processed_sentence])[0][0]


def analyze_emojis(sentence):
    emojis_to_values = {
        ')': 1,
        ':P': 1,
        '}': 1,
        '(': -1,
        '{': -1,
    }
    total = 0
    for emoji, value in emojis_to_values.items():
        total += (len(sentence.split(emoji)) - 1) * value

    return total * 0.2


def analyze_russian_sentence(sentence, nn_type):
    emojis_offset = analyze_emojis(sentence)

    sentence = CSVParseRU.stemmatize_single_sentence(sentence)
    processed_sentence = NNRussian.process_single_sentence(sentence)

    negative_result = analyze_sentence_negative(processed_sentence, nn_type) + emojis_offset
    positive_result = analyze_sentence_positive(processed_sentence, nn_type) + emojis_offset

    negative = negative_result > 0.5
    positive = positive_result > 0.5

    if negative and positive:
        result = 'Невозможно определить (позитивный и негативный)'
    elif positive:
        result = "Позитивный"
    elif negative:
        result = 'Негативный'
    else:
        result = 'Нейтральный'

    return result


def analyze_english_sentence(sentence, nn_type):
    name_to_model = dict(
        cnn=NNEnglish.convolutional_model,
        rnn=NNEnglish.recurrent_model,
        cnn_rnn=NNEnglish.convolutional_recurrent_model
    )

    model = name_to_model[nn_type]
    label = NNEnglish.predict(
        model, sentence, include_neutral=True
    )['label']

    label_to_result = {
        NNEnglish.POSITIVE: 'Позитивный',
        NNEnglish.NEUTRAL: 'Нейтральный',
        NNEnglish.NEGATIVE: 'Негативный'
    }

    return label_to_result[label]

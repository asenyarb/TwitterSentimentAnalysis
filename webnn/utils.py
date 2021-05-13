from tensorflow.keras.models import load_model

from csv_parse.csv_parse import stemmatize_single_sentence
from nn.nn import process_single_sentence

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


def analyze_sentence(sentence, nn_type):
    emojis_offset = analyze_emojis(sentence)

    sentence = stemmatize_single_sentence(sentence)
    processed_sentence = process_single_sentence(sentence)  # still not np.array

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

from tensorflow.keras.models import load_model

from nn.nn import process_single_sentence

nn_types = {
    'cnn': 'Свёрточная',
    'rnn': 'Рекуррентная',
    'cnn_rnn': 'Свёрточная и Рекуррентная параллельно'
}


def analyze_sentence_positive(processed_sentence, nn_type):
    model = load_model(f'nn/positives_nn_{nn_type}')
    return model.predict(processed_sentence)


def analyze_sentence_negative(processed_sentence, nn_type):
    model = load_model(f'nn/negatives_nn_{nn_type}')
    return model.predict(processed_sentence)


def analyze_sentence(sentence, nn_type):
    processed_sentence = process_single_sentence(sentence)  # still not np.array

    #negative_result = analyze_sentence_negative(processed_sentence, nn_type)
    #positive_result = analyze_sentence_positive(processed_sentence, nn_type)
    import time
    time.sleep(5)

    negative = False
    positive = True

    if negative and positive:
        result = 'Невозможно определить (позитивный и негативный)'
    elif positive:
        result = "Позитивный"
    elif negative:
        result = 'Негативный'
    else:
        result = 'Нейтральный'

    return result

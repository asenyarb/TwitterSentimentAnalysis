from django.shortcuts import render
from django.views import View
from webnn.utils import analyze_russian_sentence, analyze_english_sentence, nn_types
from nn.nn import NNRussian


class IndexView(View):
    def get(self, request, *args, **kwargs):
        print('GET')
        return render(
            request, 'index.html',
            context={
                'result': None,
                'error': None,
                'nn_types': list(nn_types.items()),
                'sentence': '',
                'max_words': NNRussian.matrix_rows_n,
                'max_word_symbols': NNRussian.matrix_cols_n
            }
        )

    def post(self, request, *args, **kwargs):
        sentence = request.POST['sentence']
        nn_type = request.POST['nn_type'] if 'nn_type' in request.POST else None
        lang = 'en'

        error = None
        result = None

        if nn_type is None:
            error = 'Выберите тип нейронной сети'

        if sentence and not error:
            try:
                if lang == 'ru':
                    result = analyze_russian_sentence(sentence, nn_type)
                elif lang == 'en':
                    result = analyze_english_sentence(sentence, nn_type)
                else:
                    error = 'Неизвестный язык'
            except Exception as e:
                print(e)
                error = 'Ошибка при анализе'
        elif not sentence:
            error = 'Выражение не должно быть пустым'

        return render(
            request, 'index.html',
            context={
                'result': result,
                'error': error,
                'nn_types': list(nn_types.items()),
                'sentence': sentence,
                'max_words': NNRussian.matrix_rows_n,
                'max_word_symbols': NNRussian.matrix_cols_n
            }
        )

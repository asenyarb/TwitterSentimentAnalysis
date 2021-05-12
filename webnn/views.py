from django.shortcuts import render
from django.views import View
from webnn.utils import analyze_sentence, nn_types
from nn.nn import matrix_rows_n, matrix_cols_n


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
                'max_words': matrix_rows_n,
                'max_word_symbols': matrix_cols_n
            }
        )

    def post(self, request, *args, **kwargs):
        sentence = request.POST['sentence']
        nn_type = request.POST['nn_type'] if 'nn_type' in request.POST else None

        error = None
        result = None

        if nn_type is None:
            error = 'Выберите тип нейронной сети'

        if sentence and not error:
            try:
                result = analyze_sentence(sentence, nn_type)
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
                'max_words': matrix_rows_n,
                'max_word_symbols': matrix_cols_n
            }
        )

import csv
import json
import subprocess


class CSVParseRU:
    @staticmethod
    def apply_mystem(sentence):
        command = "./csv_parse/mystem -c --format json".split(' ')
        sb = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        s = sb.communicate(sentence.encode())[0].decode().strip().replace('\n', ',')

        d = json.loads(s)
        out = ''
        for el in d:
            out += ('analysis' in el and el['analysis'] and el['analysis'][0]['lex']) or el['text']
        return out.strip()

    @classmethod
    def process_sentences_with_mystem(
        cls, in_file_name, new_tweets_txt_filepath, csv_file_tweet_text_column_index=3, step=50000
    ):
        lines_delimiter = " :::::::::: "
        with open(in_file_name, 'r') as file_r, open(new_tweets_txt_filepath, 'w') as file_w:
            c_r = csv.reader(file_r, delimiter=';')
            run = True
            while run:
                lines_set = []
                for _ in range(step):
                    try:
                        lines_set.append(next(c_r))
                    except StopIteration:
                        run = False
                        break

                text_lines_combined = lines_delimiter.join(
                    map(
                        lambda row: row[csv_file_tweet_text_column_index].replace('\n', '').replace('\\n', ''),
                        lines_set
                    )
                )
                out = cls.apply_mystem(text_lines_combined)

                lines_parsed = out.split(lines_delimiter)

                file_w.write('\n'.join(map(str, lines_parsed)))

    @classmethod
    def stemmatize_single_sentence(cls, sentence):
        sentence = sentence.replace('\n', '').replace('\\n', '')
        return cls.apply_mystem(sentence)


def parse_csvs_ru():
    print('1.')
    CSVParseRU.process_sentences_with_mystem('./data/neutral.csv', './data/neutral_parsed.csv')
    print('1.1')
    CSVParseRU.process_sentences_with_mystem('./data/negative.csv', './data/negative_parsed.csv')
    print('1.2')
    CSVParseRU.process_sentences_with_mystem('./data/positive.csv', './data/positive_parsed.csv')
    print('1.3')


if __name__ == '__main__':
    parse_csvs_ru()

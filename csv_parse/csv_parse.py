import csv
import json
import subprocess
from operator import itemgetter


def process_sentences_with_mystem(in_file_name, out_file_name, csv_file_tweet_text_column_index=3, step=50000):
    lines_delimiter = " :::::::::: "
    with open(in_file_name, 'r') as file_r, open(out_file_name, 'w') as file_w:
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
            command = "./csv_parse/mystem -c --format json".split(' ')
            sb = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            s = sb.communicate(text_lines_combined.encode())[0].decode().strip().replace('\n', ',')

            d = json.loads(s)

            out = ''
            for el in d:
                out += ('analysis' in el and el['analysis'] and el['analysis'][0]['lex']) or el['text']
            out = out.strip()

            lines_parsed = out.split(lines_delimiter)

            for csv_row, tweet_text_parsed in zip(lines_set, lines_parsed):
                csv_row[csv_file_tweet_text_column_index] = tweet_text_parsed

            c_v = csv.writer(file_w, delimiter=';')
            c_v.writerows(lines_set)


def stemmatize_single_sentence(sentence):
    command = "./csv_parse/mystem -c --format json".split(' ')
    sentence = sentence.replace('\n', '').replace('\\n', '')
    sb = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    s = sb.communicate(sentence.encode())[0].decode().strip().replace('\n', ',')

    d = json.loads(s)
    out = ''
    for el in d:
        out += ('analysis' in el and el['analysis'] and el['analysis'][0]['lex']) or el['text']
    out = out.strip()
    return out


def create_tweets_text_files(
    tweets_csv_path, new_tweets_txt_filepath,
    column_tweet_index
):
    def move_from_csv_to_txt(in_filepath, out_filepath):
        with open(in_filepath, 'r') as csv_f, open(out_filepath, 'w') as txt_f:
            c_r = csv.reader(csv_f, delimiter=';')
            csv_rows = list(c_r)
            tweets_list = list(map(itemgetter(column_tweet_index), csv_rows))
            txt_f.write('\n'.join(tweets_list))

    move_from_csv_to_txt(tweets_csv_path, new_tweets_txt_filepath)


if __name__ == '__main__':
    print('1.')
    process_sentences_with_mystem('./data/neutral.csv', './data/neutral_parsed.csv')
    print('1.1')
    process_sentences_with_mystem('./data/negative.csv', './data/negative_parsed.csv')
    print('1.2')
    process_sentences_with_mystem('./data/positive.csv', './data/positive_parsed.csv')
    print('1.3')

    print('2.')
    create_tweets_text_files('./data/neutral_parsed.csv', './data/neutral_tweets_list.txt', 3)
    print('2.1')
    create_tweets_text_files('./data/positive_parsed.csv', './data/positive_tweets_list.txt', 3)
    print('2.2')
    create_tweets_text_files('./data/negative_parsed.csv', './data/negative_tweets_list.txt', 3)
    print('2.3')

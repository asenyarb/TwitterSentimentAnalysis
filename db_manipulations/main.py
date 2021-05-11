from operator import itemgetter
from collections import namedtuple
import csv
import psycopg2
import json


def get_data_as_named_tuples(cursor):
    column_names = [col[0] for col in cursor.description]
    Result = namedtuple('Result', column_names)
    print('fetching all')
    f = cursor.fetchall()
    print('fetched')
    return [Result(*row) for row in f]


def get_data_as_dict(cursor):
    column_names = list(map(itemgetter(0), cursor.description))
    c = cursor.fetchall()
    return [
        {
            column_name: value for column_name, value in zip(column_names, row)
        } for row in c
    ]


def move_from_db_to_csv(filepath, query):
    connection = psycopg2.connect(
        host="localhost",
        database="deeplearningdb",
        user="deeplearninguser",
        password="deeplearningpwd",
        port=5433
    )

    with connection.cursor() as cursor:
        print('cursor created')
        query = query
        print('query created')
        cursor.execute(query)
        print('execute done')
        print('getting data..')
        data = get_data_as_dict(cursor)
        print('data done')

    for d in data:
        d['ttext'] = d['ttext'].replace('\n', '').replace('\t', '').replace('\r', '')

    with open(filepath, 'w') as f:
        writer = csv.DictWriter(f, data[0].keys(), delimiter=';')
        writer.writerows(data)


if __name__ == '__main__':
    move_from_db_to_csv(
        'data/neutral.csv',
        'select * from deeplearningdb.sentiment where ttype=2 and ttext is not null limit 120000;'
    )
    move_from_db_to_csv(
        'data/negative.csv',
        'select * from deeplearningdb.sentiment where ttype=-1 and ttext is not null limit 120000;'
    )
    move_from_db_to_csv(
        'data/positive.csv',
        'select * from deeplearningdb.sentiment where ttype=1 and ttext is not null limit 120000;'
    )

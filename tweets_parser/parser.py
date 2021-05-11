from parser_utils import repeat_action
import requests
import os


def get_old_posts():
    pass

def save_new_posts():
    pass


@repeat_action(1)
def collect_new_posts():
    auth_token = os.environ.get('TWITTER_API_AUTH_TOKEN', None)
    if not auth_token:
        raise ValueError('auth token should be set')
    tweets_data = requests.get(
        "https://api.twitter.com/2/tweets/search/recent?query=lang:ru",
        headers={
            'Authorization': f'Bearer {auth_token}'
        }
    ).json()

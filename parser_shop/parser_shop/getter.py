import requests


def get_content(url):
    if url.__contains__('LG'):
        flag = 'LG'
    elif url.__contains__('SAMSUNG'):
        flag = 'SAMSUNG'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.encoding
    return [flag, response]

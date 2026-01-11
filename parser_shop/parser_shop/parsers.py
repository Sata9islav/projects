import re
import bs4


def normalize_price(price):
    price = re.sub(r'[^0-9]', '', price)
    return price


def parse_eldorado(response):
    price_eldorado = {}
    soup = bs4.BeautifulSoup(response[1].text, 'lxml')
    for tag in soup.find_all('a'):
        if response[0] == 'SAMSUNG':
            if 'Ultra HD (4K) LED телевизор' in tag.text:
                index = tag.text.find('S'.upper())
                model = tag.text[index:]
                tag_price = tag.find_next('span')
                price_model = normalize_price(tag_price.text)
                price_eldorado[model] = price_model
            elif 'Ultra HD (4K) QLED телевизор' in tag.text:
                index = tag.text.find('S'.upper())
                model = tag.text[index:]
                tag_price = tag.find_next('span')
                price_model = normalize_price(tag_price.text)
                price_eldorado[model] = price_model
        elif response[0] == 'LG':
            if 'Ultra HD (4K) LED телевизор' in tag.text:
                index = tag.text.find('G'.upper())
                model = tag.text[index - 1:]
                tag_price = tag.find_next('span')
                price_model = normalize_price(tag_price.text)
                price_eldorado[model] = price_model
            elif 'Ultra HD (4K) OLED телевизор' in tag.text:
                index = tag.text.find('G'.upper())
                model = tag.text[index - 1:]
                tag_price = tag.find_next('span')
                price_model = normalize_price(tag_price.text)
                price_eldorado[model] = price_model
    return price_eldorado

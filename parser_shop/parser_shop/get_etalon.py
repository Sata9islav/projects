from parser_shop import getter, parsers
from parser_shop.engine import samsung_eldorado, lg_eldorado


price = {}


def get_price():
    for cnt in range(1, 5):
        target_samsung = samsung_eldorado + str(cnt)
        target_lg = lg_eldorado + str(cnt)
        content_samsung = getter.get_content(target_samsung)
        content_lg = getter.get_content(target_lg)
        price_samsung = parsers.parse_eldorado(content_samsung)
        price_lg = parsers.parse_eldorado(content_lg)
        price.update(price_samsung)
        price.update(price_lg)
    return price

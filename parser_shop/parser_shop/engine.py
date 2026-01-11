from parser_shop import getter, parsers
import openpyxl
from pathlib import Path


samsung_eldorado = 'https://www.eldorado.ru/c/televizory/b/SAMSUNG/?page='
lg_eldorado = 'https://www.eldorado.ru/c/televizory/b/LG/?page='


def construct_etalon():
    etalon_dict = {}
    current_dir = Path.cwd()
    base_index = (str(current_dir).find('ParserShop'))  # index of last symbol in base path
    base_path = str(current_dir)[0:base_index]
    target_path = Path(base_path) / 'ParserShop' / 'parser_shop' / 'etalon_price' / 'etalon_price.xlsx'
    etalon_price = openpyxl.load_workbook(target_path)
    sheet_ranges = etalon_price['Sheet']
    column_model = sheet_ranges['A']
    column_price = sheet_ranges['B']
    for index in range(len(column_model)):
        etalon_dict[column_model[index].value] = column_price[index].value
    return etalon_dict


def get_current_price():
    eldorado = {}
    for cnt in range(1, 4):
        target_samsung = samsung_eldorado + str(cnt)
        target_lg = lg_eldorado + str(cnt)
        content_samsung = getter.get_content(target_samsung)
        content_lg = getter.get_content(target_lg)
        price_samsung = parsers.parse_eldorado(content_samsung)
        price_lg = parsers.parse_eldorado(content_lg)
        eldorado.update(price_samsung)
        eldorado.update(price_lg)
    return eldorado


def get_comparison():
    final_dict = {}
    price_now = get_current_price()
    etalon_dict = construct_etalon()
    for key, value in price_now.items():
        if key in etalon_dict and etalon_dict[key]:
            if int(price_now[key]) < int(etalon_dict[key]):
                final_dict[key] = ' now:' + price_now[key] + ' old:' + etalon_dict[key]
    return final_dict

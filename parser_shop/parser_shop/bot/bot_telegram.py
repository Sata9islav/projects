from pathlib import Path
from parser_shop import engine, get_etalon
from os import getenv
import telebot
import time
import schedule
import threading
import openpyxl


API_TOKEN = getenv("TELEGRAM_API_TOKEN")
bot = telebot.TeleBot(API_TOKEN)


def cut_message(data):
    text = ''
    now_dict = {}
    for key, value in data.items():
        text += str(key)
        text += '-'
        text += str(value)
        text += '\n'
    if len(text) > 4096:
        cnt = 1
        for x in range(0, len(text), 4096):
            now_dict[cnt] = text[x:x + 4096]
            cnt += 1
        return now_dict
    else:
        return text


@bot.message_handler(commands=['run'])
def send_comprasion(message):
    result_comparison = engine.get_comparison()
    if len(result_comparison) != 0:
        format_etalon = cut_message(result_comparison)
        if type(format_etalon) == dict and len(format_etalon) != 0:
            cnt = 1
            for key, value in format_etalon.items():
                bot.send_message(message.chat.id, format_etalon[cnt])
        elif type(format_etalon) == str:
            bot.send_message(message.chat.id, format_etalon)
    elif len(result_comparison) == 0:
        bot.send_message(message.chat.id, 'No significant current changes')


@bot.message_handler(commands=['now'])
def send_current_price(message):
    current_price = engine.get_current_price()
    format_etalon = cut_message(current_price)
    if type(format_etalon) == dict:
        cnt = 1
        for key, value in format_etalon.items():
            bot.send_message(message.chat.id, format_etalon[cnt])
            cnt += 1
    elif type(format_etalon) == str:
        bot.send_message(message.chat.id, format_etalon)


@bot.message_handler(commands=['default'])
def send_default_price(message):
    etalon = engine.construct_etalon()
    format_etalon = cut_message(etalon)
    if type(format_etalon) == dict:
        cnt = 1
        for key, value in format_etalon.items():
            bot.send_message(message.chat.id, format_etalon[cnt])
            cnt += 1
    elif type(format_etalon) == str:
        bot.send_message(message.chat.id, format_etalon)


@bot.message_handler(content_types=['text'])
# @bot.message_handler(func=lambda message: True)
def get_text_message(message):
    if message.text == 'help':
        bot.send_message(message.from_user.id,
                         '#################################\n\n'
                         '"/run" - Runs the parser and returns the result of the comparison')
        bot.send_message(message.from_user.id,
                         '"/default" - Returns the default prices')
        bot.send_message(message.from_user.id,
                         '"/now" - Returns current prices\n\n'
                         '#################################')
    else:
        bot.send_message(message.from_user.id,
                         '#################################\n\n'
                         'To use the bot, send "/run"\n\n'
                         'To get help on the bot, send "help"\n\n'
                         '#################################')
    # print(message.from_user.id)
    # print(message.message_id)
    # print(message.chat.id)


def auto_send_comparison():
    result_comparison = engine.get_comparison()
    if len(result_comparison) != 0:
        format_etalon = cut_message(result_comparison)
        if type(format_etalon) == dict and len(format_etalon) != 0:
            cnt = 1
            for key, value in format_etalon.items():
                bot.send_message(407731974, format_etalon[cnt])
        elif type(format_etalon) == str:
            bot.send_message(407731974, format_etalon)
    elif len(result_comparison) == 0:
        bot.send_message(407731974, 'No significant current changes')


def auto_update_defprice():
    current_dir = Path.cwd()
    base_index = (str(current_dir).find('ParserShop'))  # index of last symbol in base path
    base_path = str(current_dir)[0:base_index]
    target_path = Path(base_path) / 'ParserShop' / 'parser_shop' / 'etalon_price' / 'etalon_price.xlsx'
    cnt = 1
    etalon_wb = openpyxl.Workbook()
    sheet = etalon_wb.active
    etalon_price = get_etalon.get_price()
    for key, value in etalon_price.items():
        index_a = 'A' + str(cnt)
        index_b = 'B' + str(cnt)
        column1 = sheet[index_a]
        column1.value = str(key)
        column2 = sheet[index_b]
        column2.value = str(value)
        cnt += 1
    etalon_wb.save(target_path)


schedule.every().day.at("00:30").do(auto_send_comparison)
schedule.every().day.at("05:30").do(auto_send_comparison)
schedule.every().monday.at("15:00").do(auto_update_defprice)


def go_demon():
    while True:
        schedule.run_pending()
        time.sleep(1)


second_process = threading.Thread(target=go_demon, name="task")
second_process.start()


bot.polling(none_stop=True, interval=0)

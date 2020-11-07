"""
URL: t.me/ukr_stt_bot
"""

import warnings

warnings.simplefilter('ignore')

import os
import logging
import ffmpeg
import torch

from os import remove
from os.path import dirname

import telebot
from uuid import uuid4
from utils import init_jit_model, read_batch, prepare_model_input

TOKEN = os.environ['TOKEN']

if not TOKEN:
    print('You must set the TOKEN environment variable')
    exit(1)

START_MSG = '''Вітання!

Цей бот створений для тестування перекладу українських аудіозаписів в текст.

Група для обговорення: https://t.me/speech_recognition_uk'''

FIRST_STEP = '''Використовувати бота просто: надішліть аудіоповідомлення і чекайте відповіді'''

device = torch.device('cpu')

jit_model = dirname(__file__) + '/model/ua_v1_jit.model'
model, decoder = init_jit_model(jit_model, device=device)

bot = telebot.TeleBot(TOKEN, parse_mode=None)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, START_MSG)
    bot.reply_to(message, FIRST_STEP)


@bot.message_handler(content_types=['voice'])
def process_voice_message(message):
    # download the recording
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # save the recording on the disk
    uuid = uuid4()
    filename = dirname(__file__) + f'/recordings/{uuid}.ogg'
    with open(filename, 'wb') as f:
        f.write(downloaded_file)

    # convert OGG to WAV
    wav_filename = dirname(__file__) + f'/recordings/{uuid}.wav'
    _, err = (
        ffmpeg
            .input(filename)
            .output(wav_filename, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=False)
    )
    if err is not None:
        bot.reply_to(message, 'Помилка...')
        return

    # do the recognition
    model_input = prepare_model_input(read_batch([wav_filename]), device=device)
    output = model(model_input)
    for it in output:
        # get the recognized text
        text = decoder(it.cpu())

        # no results
        if not text:
            bot.reply_to(message, 'Я не зміг розпізнати 😢')
        else:
            # send the recognized text
            bot.reply_to(message, text)

    # remove the original recording
    remove(filename)

    # remove WAV file
    remove(wav_filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    bot.polling()

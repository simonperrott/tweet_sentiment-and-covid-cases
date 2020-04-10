from bs4 import BeautifulSoup
import emoji
from data import emoticons


def count_exclamations(text):
    return text.count('!')


def __transfrom_emojis(text):
    text = emoji.demojize(text)
    text = text.replace(':', '')  # done after emoticons have been replaced
    return text


def __transfrom_emoticons_1(words):
    words = [emoticons.Emoticons_dict[w] if w in emoticons.Emoticons_dict else w for w in words]
    return words


def __transfrom_emoticons_2(text):
    for k, v in emoticons.Emoticons_dict.items():
        text = text.replace(k, v)
    return text


def clean_text(text):
    # remove html
    text = BeautifulSoup(text, 'html.parser').text

    # remove hashtag character
    text = text.replace('#', '')

    # remove words with inks & mentions
    words = text.split()
    substrings_non_grata = ['http', '@']
    words_grata = [w for w in words if not any(ss in w for ss in substrings_non_grata)]

    # rewrite emoticons
    words_grata = __transfrom_emoticons_1(words_grata)

    # rewrite emojis
    text = __transfrom_emojis(' '.join(words_grata))

    return text.lower()

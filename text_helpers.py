from bs4 import BeautifulSoup
import emoji
from data import emoticons
from textblob import TextBlob


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


def correct_misspelling(words):
    corrected_words = [str(TextBlob(w).correct()) for w in words]
    return corrected_words


def clean_text(text):

    # remove words with links, hashtags & mentions
    words = text.split()
    substrings_non_grata = ['http', '@', '#']
    clean_words = [w for w in words if not any(ss in w for ss in substrings_non_grata)]

    # rewrite emoticons
    clean_words = __transfrom_emoticons_1(clean_words)

    # remove html
    clean_text = BeautifulSoup(' '.join(clean_words), 'html.parser').text

    # rewrite emojis
    clean_text = __transfrom_emojis(clean_text)

    # correct spelling
    # clean_words = correct_misspelling(clean_words)

    return clean_text.lower()

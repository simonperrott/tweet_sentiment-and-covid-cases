import unittest
import text_helpers


class TestHelpers(unittest.TestCase):

    text = '<html> @simon says touch your #nose :) https://aiohdiaish.com http://twitpic.com/2y1zl #song #fun </html>'

    def test_hyperlinks_are_removed(self):
        output = text_helpers.clean_text(self.text)
        output_words = output.split()
        self.assertFalse('http' in output_words, 'Hyperlinksstill present')

    def test_html_is_removed(self):
        output = text_helpers.clean_text(self.text)
        self.assertFalse(any(x in output for x in ['<', '>', '/>']), 'HTML still present')

    def test_hashtags_are_removed(self):
        output = text_helpers.clean_text(self.text)
        self.assertFalse('#' in output, 'hashtags still present')

    def test_emoticons_are_removed(self):
        output = text_helpers.clean_text(self.text)
        self.assertFalse(':)' in output, 'emoticons still present')

    def test_links_are_removed(self):
        output = text_helpers.clean_text(self.text)
        self.assertFalse('http' in output, 'links still present')

if __name__ == '__main__':
    unittest.main()
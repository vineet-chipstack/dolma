import random

from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger
import re
import os
cur_path = os.path.dirname(os.path.realpath(__file__))


def keywords_in_text(keywords, text):
    # Compile a regular expression pattern from the list of keywords
    pattern = re.compile('|'.join(keywords), flags=re.IGNORECASE)

    # Search for the pattern in the text
    matches = pattern.findall(text)

    # Return True if any match is found, otherwise False
    return bool(matches)


def read_keywords_from_file(file_path):
    keywords = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and newline characters from each line
            keyword = line.strip()
            keywords.append(keyword)
    return keywords


keywords = read_keywords_from_file(os.path.join(cur_path, 'keywords.txt'))


@add_tagger("chipdesign_keywords")
class ChipDesignKeywordTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        # first, we generate a random number
        score = int(keywords_in_text(keywords, doc.text))

        # we assign the random score to a span that
        # covers the entire document
        span = Span(
            start=0,
            end=len(doc.text),
            type="document",
            score=score
        )

        # we return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=[span])


allowed_extensions = [".v", ".sv", ".vh", ".svh", ".vhd", ".vhdl",
                      ".vlg", ".vlog", ".vqm", ".vq", ".vqf", ".vqif", ".vqtf", ".vst"]


@add_tagger("chipdesign_type")
class ChipDesignTypeTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        # first, we generate a random number
        score = int(doc.metadata["ext"] in allowed_extensions)

        # we assign the random score to a span that
        # covers the entire document
        span = Span(
            start=0,
            end=len(doc.text),
            type="document",
            score=score
        )

        # we return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=[span])

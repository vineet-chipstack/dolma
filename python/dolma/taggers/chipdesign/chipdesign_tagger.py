import random

from ...core.data_types import DocResult, Document, Span
from ...core.taggers import BaseTagger
import re
import os
from ...core.registry import TaggerRegistry

cur_path = os.path.dirname(os.path.realpath(__file__))


def keywords_in_text(keywords, text):
    # Compile a regular expression pattern from the list of keywords
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word)
                         for word in keywords) + r')\b', flags=re.IGNORECASE)

    # # Search for the pattern in the text
    # matches = pattern.findall(text)

    # Finding all matches and their indices
    matches = [(match.group(), match.start())
               for match in pattern.finditer(text)]

    # Return True if any match is found, otherwise False
    return matches


def read_keywords_from_file(file_path):
    keywords = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and newline characters from each line
            keyword = line.strip()
            keywords.append(keyword)
    return keywords


keywords = read_keywords_from_file(os.path.join(cur_path, 'keywords.txt'))


# @add_tagger("chipdesign_keywords")
@TaggerRegistry.add("chipdesign_keywords")
class ChipDesignKeywordTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        # first, we generate a random number
        matches = keywords_in_text(keywords, doc.text)
        score = len(matches)

        spans = []
        # we assign the random score to a span that
        # covers the entire document
        for m in matches:
            span = Span(
                start=m[1],
                end=m[1] + len(m[0]),
                type=f"{m[0]}",
                score=1
            )
            spans.append(span)
        span = Span(
            start=0,
            end=len(doc.text),
            type="document",
            score=score
        )
        spans.append(span)

        # we return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=spans)


allowed_extensions = [".v", ".sv", ".vh", ".svh", ".vhd", ".vhdl",
                      ".vlg", ".vlog", ".vqm", ".vq", ".vqf", ".vqif", ".vqtf", ".vst"]


# @add_tagger("chipdesign_type")
@TaggerRegistry.add("chipdesign_type")
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

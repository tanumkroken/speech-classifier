#  Copyright (c) 2021 by Ole Christian Astrup. All rights reserved.  Licensed under MIT
#   license.  See LICENSE in the project root for license information.
#
import spacy
import package_deps
from spacy.lang.nb.examples import sentences
import graphviz, deplacy

print spacy.de

nlp = spacy.load("nb_core_news_sm")
doc=nlp("Hvor er Ole")
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)



deplacy.render(doc)
deplacy.serve(doc,port=None)

graphviz.Source(deplacy.dot(doc))
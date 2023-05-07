# coding=utf8
from xml.dom.expatbuilder import parseString
from reynir import Greynir

g = Greynir()
#sent = g.parse_single("Talið er að hún hafi verið á leiðinni til Íslands ásamt öðrum álftum þegar slysið varð.")

fulltext = "Hún réð sig til vinnu á gúmmíbát."
"""Möguleg viðbrögð rædd.
Merkingum mögulega ábótavant.
Forsetinn með fiskabindi.
Mannlíf í myndum.
Þvert á evrópska bílaframleiðendur.
Fjármálaráðuneytið bíður eftir RÚV.
Lokahátíð Ólympíuleikanna í Ríó.
Mjög skemmtilegt verkefni.
Áttu þér uppáhaldshúðvörur?"""

d = g.parse(fulltext)
print("{0} sentences were parsed".format(d["num_parsed"]))
for sent in d["sentences"]:
    print("The parse tree for '{0}' is:\n{1}"
        .format(
            sent.tidy_text,
            "[Null]" if sent.tree is None else sent.tree.flat
        )
    )
print("Done.")
print("Number of sentences: {}".format(d["num_sentences"]))
print("Thereof successfully parsed: {}".format(d["num_parsed"]))
print("Average ambiguity factor: {}".format(d["ambiguity"]))
print("Parse time: {}".format(d["parse_time"]))
print("Reduction time: {}".format(d["reduce_time"]))


'''
sents = fulltext.split("\n")

i = 1
for sent in sents:
    if sent == "":
        break
    sent = sent + "."
    parsedSent = g.parse_single(sent)
    print(sent)
    if(parsedSent.err_index is None):
        print("Sent " + str(i) + " is ambiguous: " + str(parsedSent._tree.is_ambiguous))
    else:
        print("Could not parse sentence. Error index: " + str(parsedSent.err_index))
    i += 1

text1 = "Umframfiskur ratar á diska fátæka fólksins"
text2 = "Hún réð sig til vinnu á gúmmíbát"

#sent = g.parse_single(text2)
job = g.submit(text2)
for sent in job:
    if sent.parse():
        # print(sent.tree)
        print("Sentence parsed - skipping showing the whole tree.")
    else:
        print("Error at index {}".format(sent.err_index))
num_sentences = job.num_sentences   # Total number of sentences
num_parsed = job.num_parsed         # Thereof successfully parsed
ambiguity = job.ambiguity           # Average ambiguity factor
parse_time = job.parse_time         # Elapsed time since job was created
reduction_time = job.reduce_time    # Reduction time
print("Done.")
print("Number of sentences: {}".format(num_sentences))
print("Thereof successfully parsed: {}".format(num_parsed))
print("Average ambiguity factor: {}".format(ambiguity))
print("Parse time: {}".format(parse_time))
print("Reduction time: {}".format(reduction_time))

#print(sent.tree.view)
#print("Is it?: " + str(sent._tree.is_ambiguous))'''
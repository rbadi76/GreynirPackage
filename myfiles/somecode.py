# coding=utf8
from xml.dom.expatbuilder import parseString
from reynir import Greynir

g = Greynir()
#sent = g.parse_single("Talið er að hún hafi verið á leiðinni til Íslands ásamt öðrum álftum þegar slysið varð.")

fulltext = "Merki um stríðsglæpi Rússa sé að finna í nánast hverri einustu borg sem þeir hafa yfirgefið, segir Óskar Hallgrímsson ljósmyndari í Kænugarði. „Maður er alveg farinn að finna fyrir að fyrir að taugakerfið  er ekkert sérstaklega gott eftir að hafa verið með það útþanið í tæplega þrjá mánuði. En ég og kona mín erum ágætir félagar og styðjum hvort annað vel í þessu“ sagði Óskar í morgunþætti Rásar tvö í morgun. „Stríðið skilur eftir sig svakalegt sár hjá fólki og reiðin er orðin mikil. Rússar eru hættir að heyja stríð eins og maður myndi halda að hefðbundið stríð væri.  Nú ráðast þeir á borgarleg skotmörk og þær borgir sem Rússar hafa hertekið og yfirgefið eru í rúst. Það virðist vera merki um stríðsglæpi og nauðganir á nánast hverjum einasta stað sem þeir hafa verið ár.“"

sents = fulltext.split(".")

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

print("Done")

#print(sent.tree.view)
#print("Is it?: " + str(sent._tree.is_ambiguous))
WITH '

%%sparql%%

' AS sparql

CALL n10s.rdf.import.fetch(
  'https://query.wikidata.org/sparql?query='+ apoc.text.urlencode(sparql),
  'Turtle' ,
  { headerParams: { Accept: "application/x-turtle" } }
)
YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo
RETURN terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo
# PDBs during the Cuban Missile Crisis (10/16/1962 - 10/28/1962)
curl "https://api.foiarchive.org/docs?corpus=eq.pdb&authored=gte.1962-10-16&authored=lt.1962-10-28"
# same query with columns limited
curl "https://api.foiarchive.org/docs?select=doc_id,authored,source&\
corpus=eq.pdb&authored=gte.1962-10-16&authored=lt.1962-10-28"
# same query formatted nicely
curl "https://api.foiarchive.org/docs?select=doc_id,authored,source&\
corpus=eq.pdb&authored=gte.1962-10-16&authored=lt.1962-10-28" | \
python -m json.tool
# PDB during crisis that reference Khrushchev by name in text
curl "https://api.foiarchive.org/docs?select=doc_id,authored,source&\
corpus=eq.pdb&authored=gte.1962-10-16&authored=lt.1962-10-28&\
full_text=wfts.khrushchev"
# for more information https://postgrest.org/en/stable/
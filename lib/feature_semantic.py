import requests
from pathlib import Path
import joblib
from scipy import sparse
import pandas as pd
import numpy as np
from datetime import date
import networkx as nx
from collections import deque
import json
from itertools import chain

uniteresting_nodes = [
    "Q189533",
    "Q3529618",
    "Q4218455",
    "Q28640",
    "Q35120",
    "Q103940464",
    "Q488383",
    "Q20937557",
    "Q1190554",
    "Q1150070",
    "Q23958946",
    "Q3249551",
    "Q107329943",
    "Q11424100",
    "Q83492918",
    "Q24229398",
    "Q189970",
    "Q83493482",
    "Q7302601",
    "Q618779",
    "Q31464082",
    "Q1554231",
    "Q106559804",
    "Q830077",
    "Q18336849",
    "Q795052",
    "Q215627",
    "Q67518978",
    "Q46737",
    "Q104637332",
    "Q427581",
    "Q67518233",
    "Q374814",
    "Q16023913",
    "Q702269",
    "Q43229",
    "Q15401930",
    "Q7048977",
    "Q7184903",
    "Q11028",
    "Q386724",
    "Q13235160",
    "Q17537576",
    "Q100195948",
    "Q47461344",
    "Q2500638",
    "Q15621286",
    "Q286583",
    "Q184754",
    "Q4897819",
    "Q107435521",
    "Q482980",
    "Q36180",
    "Q327055",
    "Q37866906",
    "Q234460",
    "Q49848",
    "Q214339",
    "Q1914017",
    "Q15980158",
    "Q1650915",
    "Q16686448",
    "Q20826540",
    "Q217577",
    "Q4164871",
    "Q12488383",
    "Q34394",
    "Q104127086",
    "Q1048607",
    "Q9081",
    "Q61788060",
    "Q451967",
    "Q1807498",
    "Q12737077",
    "Q17232516",
    "Q8187769",
    "Q6958747",
    "Q192581",
    "Q16334298",
    "Q13420330",
    "Q16334295",
    "Q4026292",
    "Q1047113",
    "Q2169973",
    "Q852998",
    "Q108289030",
    "Q187931",
    "Q2944660",
    "Q3695082",
    "Q852998",
    "Q93288",
    "Q2585724",
    "Q928786",
    "Q3505845",
    "Q1656682",
    "Q6671777",
    "Q8171",
    "Q1914636",
    "Q5127848",
    "Q2995644",
    "Q4189293",
    "Q1999851",
    "Q4120621",
    # occupation
    "Q1786828",
    "Q1293220",
    "Q99527517",
    "Q28813620",
    "Q16887380",
    "Q98119401",
    "Q61961344",
    "Q937228",
    "Q1207505",
    "Q58415929",
    "Q26907166",
    "Q9332",
    "Q3769299",
    "Q2897903",
    "Q28877",
    "Q26720107",
    "Q53617489",
    "Q27043950",
    "Q7239",
    "Q729",
    "Q159344",
    "Q15978631",
    "Q164509",
    "Q154954",
    "Q5",
    "Q483247",
    "Q813912",
    "Q54989186",
    "Q58778",
    # religion
    "Q930933",
    "Q12558574",
    "Q595523",
    "Q2145290",
    "Q4393498",
    "Q1347367",
    "Q151885",
    "Q33104279",
    "Q131841",
    "Q49447",
    "Q55919789",
    "Q2990593",
    "Q2083958",
    "Q18593264",
    "Q36529775",
    "Q2393196",
    "Q2515887",
    "Q838948",
    "Q55979391",
    "Q36192",
    "Q1299714",
    "Q42848",
    "Q37787614",
    "Q61151961",
    "Q48324",
    "Q109810475",
    "Q251777",
    "Q5390013",
    "Q7257",
    "Q71966963",
    "Q82821",
    "Q3702971",
    "Q9174",
    "Q1530022",
    "Q29896155",
    "Q72638",
    "Q10862449",
    "Q5801290",
    "Q5067949",
    "Q203066",
    "Q4406616",
    "Q53001749",
    "Q223557",
    "Q7254446",
    "Q1412596",
    "Q8205328",
    "Q216920",
    "Q621184",
    "Q18603648",
    "Q86923152",
]

class Fetcher:
    def __init__(self):
        self.cache = {}
        self.WIKIDATA_NAMESPACE = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        
    def get_super_classes(self, qid, relation_pid="P279"):
        if qid in self.cache:
            return self.cache[qid]
        
        resp = self.session.get(self.WIKIDATA_NAMESPACE.format(qid))
        if resp.status_code != 200:
            print(f"The HTTP status is not 200 but {resp.status_code}")
            return ""
    
        try:
            doc = json.loads(resp.text)
        except json.decoder.JSONDecodeError as err:
            print(f"Error decoding: {err}")
            return ""
    
        entities = doc["entities"]
        if qid in entities:
            entity = entities[qid]
        else:
            # in case of redirection
            entity = next(iter(entities.values()))
    
        claims = entity["claims"]
        SUBCLASS_PID = relation_pid

        res = []
        if relation_pid in claims:
            res = [c["mainsnak"]["datavalue"]["value"]["id"] for c in claims[relation_pid] if "datavalue" in c["mainsnak"]]
            
        self.cache[qid] = res
        return res

    def persist(self, f):                    
        json.dump(self.cache, f)

    def restore(self, f):
        self.cache = json.load(f)


def build_network(init_qids, hop_limit, fetcher):
    if hop_limit <= 0:
        return None
    
    g = nx.DiGraph()
    
    g.add_nodes_from(q for q in init_qids)
    pending = deque((q, hop_limit) for q in init_qids)
    while len(pending) > 0:
        qid, entity_hop_limit = pending.popleft()
        superclasses = fetcher.get_super_classes(qid)
        g.add_edges_from((qid, c) for c in superclasses)
        if entity_hop_limit > 1:
            pending.extend((c, entity_hop_limit-1) for c in superclasses)
    return g
            

def get_network(fetcher, prefix, vectorizer, hop_limit):
    categories = [c[len(prefix)-1:] for c in vectorizer.classes_ if c.startswith(prefix)]
    return build_network(categories, hop_limit, fetcher)

def clean_graph(g, classification_nodes):
    g_cut = g.copy()
    g_cut.remove_edges_from(list(e for e in g_cut.edges() if e[0] in classification_nodes))
    
    g_cut_closure = nx.algorithms.dag.transitive_closure(g_cut)
    classified = set(
        chain.from_iterable(g_cut_closure.predecessors(n) for n in classification_nodes)
    )
    g_cut_closure.remove_edges_from(
        list(
            e
            for e in g_cut_closure.edges()
            if e[0] in classified and e[1] not in classification_nodes
        )
    )
    return g_cut_closure

def get_class_map(g, classification_nodes):
    return dict((id, c) for c in classification_nodes for id in g.predecessors(c))

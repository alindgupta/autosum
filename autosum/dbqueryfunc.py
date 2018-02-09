import argparse
import datetime
import multiprocessing
from typing import Tuple, NamedTuple, Iterable
from collections import namedtuple
import requests
from bs4 import BeautifulSoup

url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
esearch_base = url_base + 'esearch.fcgi?'
efetch_base = url_base + 'efetch.fcgi?'

def process_request(url: str,
                    tags: Iterable[str],
                    parser='html.parser') -> Dict[str, str]:
    contents = {}
    try:
        raw_html = requests.get(url)
        parse = BeautifulSoup(raw_html.content, parser)
        for tag in tags:
            contents[tag] = parse.find(tag).text 
    except requests.exceptions.RequestException:
        raise
    except Exception:
        raise
    return contents


class DBQuery:
    def __init__(self,
                 query: str,
                 database='pubmed',
                 retmax=500):
        
        if query == '':
            raise ValueError('Provided empty string')

        if database not in ('pubmed', 'pmc'):
            raise ValueError(f'Unknown database: {database}')
        
        self._db = database
        self._retmax = int(retmax)
        self._query = '+'.join(query.split(' '))
        self._count = 0

    def esearch(self):
        url = esearch_base + f'db={self._db}&term={self._query}&usehistory=y'
        try:
            contents = process_request(url, ('webenv', 'queryKey', 'count'))
        except Exception:
            raise
        self._count = contents['count']
        return contents
            
        
        


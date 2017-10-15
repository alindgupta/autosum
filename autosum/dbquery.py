"""
dbquery.py
~~~~~~~~~~~~~~~

This module handles text scraping using the NCBI E-utils.

"""

import functools
import datetime
from typing import Tuple
import requests
from bs4 import BeautifulSoup

# NCBI E-utils URLs
url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
esearch_base = url_base + 'esearch.fcgi?'
efetch_base = url_base + 'efetch.fcgi?'


class DBQuery:
    def __init__(self, query: str, db='pubmed'):
        """
        Initialize a DBQuery object.

        :param query: A term to search (eg. 'breast cancer')
        :param db: Database, one of 'pubmed' or 'pmc'
        :param retmax: Maximum number of results to fetch

        """
        if query == '':  # check explicitly for an empty string
            raise ValueError('Please provide a string')

        self.db = db
        self._query = '+'.join(query.split(' '))
        self._count = 0

    # IO
    @functools.lru_cache(maxsize=1)
    def esearch(self) -> Tuple[str, str, int]:
        """
        Execute a search request from the database.

        :returns: A tuple containing WebEnv, Query key and PMID/UID counts

        """
        url = esearch_base + ('db={}&term={}&usehistory=y'.
                              format(self.db, self._query))
        try:
            raw_html = requests.get(url)
            # might change to lxml parser in the future for speed
            parse = BeautifulSoup(raw_html.content, 'html.parser')
            webenv = parse.find('webenv').text
            query_key = parse.find('querykey').text
            count = parse.find('count').text
        except requests.exceptions.RequestException:
            raise
        return webenv, query_key, int(count)

    # IO
    @functools.lru_cache(maxsize=1)
    def fetch(self, filename, timeout=0.001) -> str:
        """
        Execute a fetch request from the database.

        :param timeout: Timeout parameter for Requests.get()

        :returns: The filename to which data was written

        """
        webenv, query_key, count = self.esearch()
        if self._count == 0:
            raise ValueError('Zero PMID/UID count, check query')
        retmax = 500
        retstart = 0
        filename = (self._query + datetime.date.today().strftime('%d%B%Y')
                    + '.txt')
        with open(filename, 'rb') as _file:
            while retstart < count:
                abstracts = ''
                url = (efetch_base
                       + 'db={}&retstart={}&retmax={}&WebEnv={}&query_key={}'
                       .format(self.db, retstart, retmax, webenv, query_key))
                try:
                    raw_html = requests.get(url)
                    parse = (BeautifulSoup(raw_html.content, 'html.parser')
                             .find_all('abstract'))
                    for item in parse:
                        abstracts += item.text + '\n'
                    _file.write(abstracts)
                except requests.exception.RequestException:
                    continue
                retstart += retmax
        return filename

    @property
    def count(self):
        return self._count

    @property
    def query(self):
        return self._query

    def __repr__(self):
        return '<class DBQuery, query={}>'.format(self._query)

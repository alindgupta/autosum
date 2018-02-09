"""
dbquery.py
~~~~~~~~~~~~~~~

This module handles text scraping using the NCBI E-utils.

"""

import functools
import argparse
import datetime
from typing import Tuple
import requests
from bs4 import BeautifulSoup

# NCBI E-utils URLs
url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
esearch_base = url_base + 'esearch.fcgi?'
efetch_base = url_base + 'efetch.fcgi?'


class DBQuery:
    def __init__(self, query: str, retmax=0, db='pubmed'):
        """ Initialize a DBQuery object.

        Parameters
        ----------
        query: string
            Search term (eg. 'breast cancer').
        db: string
            Database to query. Currently, only `pubmed` is supported.

        Raises
        ------
        ValueError: if an empty query is provided.

        """
        if query == '':
                raise ValueError('Please provide a string')

        self.db = db
        self._query = '+'.join(query.split(' '))
        self._count = 0
        self._retmax = retmax

    @functools.lru_cache(maxsize=1)
    def _esearch(self) -> Tuple[str, str, int]:
        """ Execute a search request from Pubmed (i.e. fetch PMID/UIDs).

        Returns
        -------
        Tuple[str, str, int] of WebEnv, Query key and PMID/UID counts.

        """
        url = esearch_base + 'db={self.db}&term={self._query}&usehistory=y'
        try:
            raw_html = requests.get(url)
            parse = BeautifulSoup(raw_html.content, 'html.parser')
            webenv = parse.find('webenv').text
            query_key = parse.find('querykey').text
            self._count = count = parse.find('count').text
        except requests.exceptions.RequestException:
            raise
        return webenv, query_key, int(count)

    def fetch(self, filename, verbose=False, timeout=0.001) -> str:
        """ Execute a fetch request from the database.

        Parameters
        ----------
        filename: str
            A filename/absolute filepath to write data to.
        verbose: bool
            Print progress bars.
        timeout: float
            Timeout (in milliseconds) for Requests.get().

        Returns
        ------
        None.

        """
        webenv, query_key, count = self._esearch()
        if self._count == 0:
            raise ValueError('Zero PMID/UID count, check query')
        retmax = self._retmax if self._retmax > 0 else 500
        retstart = 0
        filename = (self._query + datetime.date.today().strftime('%d%B%Y') + '.txt')
        with open(filename, 'a') as file_handle:
            while retstart < count:
                abstracts = ''
                url = (efetch_base
                       + (f'db={self.db}&'
                          'retstart={retstart}&'
                          'retmax={retmax}&'
                          'WebEnv={webenv}&'
                          'query_key={query_key}&'
                          'retmode=xml'))
                try:
                    raw_html = requests.get(url, timeout=timeout)
                    bs = BeautifulSoup(raw_html.content)
                    parse = bs.find_all('abstracttext')
                    for item in parse:
                        abstracts += item.text + '\n'
                        file_handle.write(abstracts)
                except requests.exceptions.RequestException:
                    pass
                except Exception:
                    raise
                retstart += retmax
        return

    @property
    def count(self):
        return self._count

    @property
    def query(self):
        return self._query

    def __repr__(self):
        return f'<class DBQuery, query={self._query}>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--query',
        '-q',
        help='Enter search terms to query Pubmed.')
    args = parser.parse_args()
    dbquery = DBQuery(args.query)
    dbquery.fetch('temp.txt')

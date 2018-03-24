# -*- coding: utf-8 -*-

"""dbquery.py

This module uses the NCBI E-utilities to retrieve abstracts
and saves them to a file. Multiple processes can be used to
parallelize web scraping.

Requires Python 3.6 or higher.

Example:
        $ python3 dbquery.py -f "summary.txt" -q "breast cancer BRCA1" -n 4

Attributes:
     DbQuery (class): Invoke the ``run`` method after initialization.
         Some queries can take a while to download due to the large number
         of studies (eg. "cancer"). It is often useful to qualify queries
         to make them less generic (eg. "breast cancer BRCA1").

Todo:
    * Command line parsing.
    * Stream instead of bulk download to save memory usage.
    * Pipe parsing and tokenization to reduce IO time.

"""
import sys
import time
import argparse
import functools
import multiprocessing as mp
from typing import Iterable, Union, Dict, List, IO
import requests
from bs4 import BeautifulSoup

assert sys.version_info >= (3, 6)

# constants for E-utilities
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
BASE_ESEARCH_URL = BASE_URL + 'esearch.fcgi?'
BASE_EFETCH_URL = BASE_URL + 'efetch.fcgi?'


def process_request(url: str, tags: Union[str, Iterable[str]]) \
    -> Dict[str, Union[str, List[str]]]:
    """ Scrape and parse xml retrieved from ``url``
    and return parsed contents within ``tags``.

    Parameters
    ----------
    url: str
    tags: str or an iterable of str
        HTML tag(s) to retrieve the contents of.

    Returns
    -------
    A dictionary of ``tags`` as keys
    and their contents (str or list of str) as values.

    Raises
    ------
    HTML and parsing errors from ``requests`` and ``BeautifulSoup`` libraries.

    """
    contents = {}
    if isinstance(tags, str):
        tags = (tags,)
    raw_html = requests.get(url)
    parse = BeautifulSoup(raw_html.content, 'html.parser')
    for tag in tags:
        data = parse.find_all(tag)
        if len(set(data)) == 1:
            contents[tag] = data[0].text
        else:
            contents[tag] = [item.text for item in data]  # keep picklable
    return contents

def esearch(query: str, db: str) -> Dict[str, Union[str, List[str]]]:
    """ Perform an E-search.
    Does not need to be called by the user, except perhaps to
    verify the count.

    Parameters
    ----------
    query: str
    db: str
        A database, one of either "pubmed" or "pmc".

    Returns
    -------
    dict with three keys: "webenv", "queryKey" and "count".
        These are required for fetching summaries.

    """
    url = BASE_ESEARCH_URL + f'db={db}&term={query}&usehistory=y'
    return process_request(url, ('webenv', 'queryKey', 'count'))

def efetch(
        db: str,
        retmax: int,
        webenv: str,
        query_key: int,
        retstart: int) \
        -> str:
    """ Perform an E-fetch. Scrape and parse summaries.
    Does not need to be called by the user.

    Parameters
    ----------
    db: str
        A database, one of "pubmed" or "pmc".
    retmax: int
        Maximum number of summaries to request at one time.
    webenv: str
        Web environment flag.
    query_key: int
        Query key flag.
    retstart: int
        If the number of summaries to retrieve is large,
        this can be used to retrieve summaries in batches.

    Returns
    -------
    str
        A concatenated list of strings as a single string.
    
    """
    url = (BASE_EFETCH_URL + (f'db={db}&'
                              f'retstart={retstart}&'
                              f'retmax={retmax}&'
                              f'WebEnv={webenv}&'
                              f'query_key={query_key}&'
                              f'retmode=xml'))
    print(url)
    contents = process_request(url, ('abstracttext',))
    return '\n'.join(contents['abstracttext'])


class DbQuery:
    def __init__(self, query):
        """ Initialize a DbQuery object.

        Parameters
        ----------
        query: str
            A query of interest, such as "breast cancer BRCA1".

        """
        if isinstance(query, list):
            self.query = '+'.join(query)
        elif isinstance(query, str):
            self.query = '+'.join(query.split(' '))
            self.db = 'pubmed'
            self.retmax = 500

    def run(self, filename: str, parallel=True) -> IO[str]:
        """ Initialize a DbQuery object.

        Parameters
        ----------
        filename: str
            A filename/filepath to write summaries to.
            Summaries are appended to a file if it exists.
        parallel: bool
            Use multiprocessing to fetch summaries in parallel.

        Returns
        -------
        A string is written to a file.

        """
        t = time.time()
        esearch_dict = esearch(self.query, self.db)
        print('Completed searching for abstracts in '
              f' {(time.time() - t):.2f} seconds.')

        # multiprocessing setup
        num_processes = 4 if parallel else 1
        pool = mp.Pool(num_processes)
        queue = list(range(0, int((esearch_dict['count'])[0]), self.retmax))

        # <queryKey><\queryKey> contains nothing if
        # the number of abstracts is small
        query_key = esearch_dict['queryKey'] if esearch_dict['queryKey'] else 1
        efetch_ = functools.partial(
            efetch,
            self.db,
            self.retmax,
            esearch_dict['webenv'],
            query_key)

        # main loop
        with open(filename, 'a') as fhandle:
            for string in pool.map(efetch_, queue):
                fhandle.write(string)
        print(f'Completed fetching in {(time.time() - t):.2f} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform an E-fetch')
    parser.add_argument('query', nargs='+', type=str)
    parser.add_argument('-f', default='')
    args = parser.parse_args()

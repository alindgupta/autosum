import argparse
import datetime
import functools
import requests
from bs4 import BeautifulSoup

url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
esearch_base = url_base + 'esearch.fcgi?'
efetch_base = url_base + 'efetch.fcgi?'


def process_request(url, tags, parser='html.parser'):
    """ Scrape a webpage and extract contents of an html tag.

    Parameters
    ----------
    url: str
    tags: str or Iterable[str]
        A list of tags to extract (eg. `div`).
    parser: a parser recognized by BeautifulSoup.

    Returns
    -------
    Dictionary containing tags as keys and parsed content as values.

    """
    contents = {}
    if isinstance(tags, str):
        tags = [tags]
    try:
        raw_html = requests.get(url, timeout=0.001)
        parse = BeautifulSoup(raw_html.content, parser)
        for tag in tags:
            contents[tag] = parse.find(tag).text
    except requests.exceptions.RequestException:
        raise
    except Exception:
        raise
    return contents


class DBQuery:
    def __init__(self, query, database='pubmed', retmax=500):
        """ Initialize a DBQuery object.

        Parameters
        ----------
        query: str
            A query (eg. 'breast cancer').
        database: str
            Database to query, one of either 'pubmed' or 'pmc'.
        retmax: int
            The maximum number of articles to scrape.

        """

        if query == '':
            raise ValueError('Provided empty string')

        if database.lower() not in ('pubmed', 'pmc'):
            raise ValueError(f'Unknown database: {database}')

        self._db = database.lower()
        self._retmax = int(retmax)
        self._query = '+'.join(query.split(' '))
        self._count = 0

    @property
    def query(self):
        return self._query

    @functools.lru_cache(maxsize=1)
    def _esearch(self):
        """ Get PMIDs/UIDs and counts.

        You should not need to call this method directly since
        `efetch` calls this method automatically.

        The results are cached so the method executes only once
        in any case.

        """
        url = esearch_base + f'db={self._db}&term={self._query}&usehistory=y'
        try:
            contents = process_request(url, ['webenv', 'queryKey', 'count'])
        except Exception:
            raise
        self._count = contents['count']
        return contents

    def efetch(self, filename=''):
        """ Scrape abstracts from open source articles and append
        to a text (.txt) file.

        Parameters
        ----------
        filename: str, optional argument
            A .txt filename. The file may or may not exist.

        Returns
        -------
        The filename to which data has been written/appended.

        """
        data = self._esearch()
        webenv = data['webenv']
        query_key = data['queryKey']
        retmax = max(self._retmax, 500)

        if not filename or not filename.endswith('.txt'):
            filename = (self._query
                        + datetime.date.today().strftime('%d%B%Y')
                        + '.txt')

        with open(filename, 'a') as file_handle:
            count = self._count
            retstart = 0
            while retstart < count:
                url = (efetch_base
                       + (f'db={self.db}&'
                          'retstart={retstart}&'
                          'retmax={retmax}&'
                          'WebEnv={webenv}&'
                          'query_key={query_key}&'
                          'retmode=xml'))
                try:
                    contents = process_request(url, 'abstracttext')
                    file_handle.write('\n'.join(contents['abstracttext']))
                except Exception:
                    raise
                retstart += retmax
        return filename

    def __repr__(self):
        return f'<class DBQuery, query={self._query}>, count={self._count}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--query',
        '-q',
        help='Enter search terms to query Pubmed.')
    args = parser.parse_args()
    dbquery = DBQuery(args.query)
    dbquery.fetch('temp.txt')

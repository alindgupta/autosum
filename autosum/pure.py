import time
import argparse
import functools
import requests
from bs4 import BeautifulSoup

# constants, should be set once and only once during execution
_db = 'pubmed'
_retmax = 5000
url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
esearch_base = url_base + 'esearch.fcgi?'
efetch_base = url_base + 'efetch.fcgi?'

def process_request(url: str, tags: Iterable[str]) -> Dict[str, str]:
    """ Connect to url, scrape data, parse xml and return parsed contents in tags """
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
            contents[tag] = (item.text for item in data)
    return contents

def esearch(query):
    url = esearch_base + f'db={self._db}&term={query}&usehistory=y'
    return process_request(url, ('webenv', 'queryKey', 'count'))

def efetch(webenv, query_key, retstart):
    url = (efetch_base + (f'db={self._db}&'
                          f'retstart={retstart}&'
                          f'retmax={retmax}&'
                          f'WebEnv={webenv}&'
                          f'query_key={query_key}&'
                          f'retmode=xml'))
    contents = process_request(url, ('abstracttext',))
    return '\n'.join(contents['abstracttext'])



def main(filename):
    parser = ArgumentParser()
    parser.add_argument('--query', '-q')
    args = parser.parser_args()
    t = time.time()
    esearched = esearch('+'.join(args.query.split(' ')))
    print(f'Completed searching for abstracts in {time.time() - t : .2f} seconds.')
    queue = range(0, esearched['count'], retmax)
    pool = mp.Pool(mp.cpu_count())
    efetch_ = functools.partial(efetch, esearch['webenv'], esearched['queryKey'])
    with open(filename, 'a') as fhandle:
        for string in pool.map(efetch_, pool)
            fhandle.write(string)
    print(f'Completed fetching in {time.time() - t : .2f} seconds.')

def test_proc_req():
    """ Test cases
    1. Empty url -> requests error
    2. Invalid url -> requests error
    3. No tags -> error 
    4. Empty tag -> return empty dict
    5. Invalid tag -> return empty dict 
    6. No internet connection -> requests error 
    7. Malformed html/xml -> 
    """
    test_ok = 0
    valid_url = 'http://www.wikipedia.com/python'
    invalid_url = 'http://www.wikipedia.com/err'
    try:
        process_request(valid_url, valid_tags)
        test_ok = 1
    except:
        pass
    assert test_ok == 0

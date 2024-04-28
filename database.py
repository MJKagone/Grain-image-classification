from tinydb import TinyDB, Query

db = TinyDB('data/db.json')
Seed = Query()

def insert(entry_type, path, set='none'):
    db.insert({'type': entry_type, 'path': path, 'set': set})


def insertTest():

    '''Test that database works.'''

    db.insert({'type': 'kaura', 'path': 'test/kaura/000_000.bpm', 'set': 'training'})
    db.insert({'type': 'kaura', 'path': 'test/kaura/000_001.bpm', 'set': 'training'})
    db.insert({'type': 'kaura', 'path': 'test/kaura/000_002.bpm', 'set': 'training'})

    db.insert({'type': 'ohra', 'path': 'test/ohra/000_000.bpm', 'set': 'training'})

    db.insert({'type': 'ruis', 'path': 'test/ruis/000_000.bpm', 'set': 'training'})
    db.insert({'type': 'ruis', 'path': 'test/ruis/000_000.bpm', 'set': 'training'})

def reset(database=db):
    '''Reset database, delete everything'''
    database.truncate()

def searchType(entry_type):
    '''
    Returns each element with specified seed type
    '''
    results = db.search(Seed.type == entry_type)
    return results



import dbm
import json


def save_json_data(dbname, key, json_var):
    try:
        db = dbm.open(dbname, 'c')
        db[key] = json.dumps(json_var)
    except Exception as e:
        print(e)
        return False
    finally:
        db.close()
    return True


def load_json_data(dbname, key):
    try:
        db = dbm.open(dbname, 'r')
        r = json.loads(db[key].decode('utf-8'))
        db.close()
    except Exception as e:
        print(e)
        return None
    return r


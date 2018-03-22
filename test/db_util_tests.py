import unittest
from db_util import save_json_data, load_json_data
import dbm


class DbTestCase(unittest.TestCase):
    def test_save_load(self):
        jdata = {'a': 1, 'b': 0}
        save_json_data('testdb', 'k', jdata)
        load_json_data('testdb', 'k')
        load_data = load_json_data('testdb', 'k')
        self.assertEqual(jdata, load_data)


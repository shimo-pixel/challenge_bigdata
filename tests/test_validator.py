import unittest

from text2query.validator import validate_query


CATALOG = {"tables": [{"source": "MYSQL", "name": "vehicle", "database": "transportation", "columns": []}]}


class ValidatorTest(unittest.TestCase):
    def test_select_is_accepted(self):
        self.assertTrue(validate_query("MYSQL", "SELECT * FROM transportation.vehicle", CATALOG).ok)

    def test_write_is_rejected(self):
        self.assertFalse(validate_query("MYSQL", "DELETE FROM transportation.vehicle", CATALOG).ok)

    def test_cross_requires_result(self):
        self.assertFalse(validate_query("MYSQL", "SELECT * FROM transportation.vehicle", CATALOG, True).ok)


if __name__ == "__main__":
    unittest.main()

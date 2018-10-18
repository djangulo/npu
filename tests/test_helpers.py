import unittest
import sys, os
import pandas as pd
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
print(BASE_DIR)

from npu import helpers

class HelpersTest(unittest.TestCase):

    def test_worksheet_names_exists(self):
        """It should find the worksheet names in the 'data' dir."""
        expected = 'worksheet_names.txt'
        msg = '\n\nThe function could not find the file named '\
              '"worksheet_names.txt" in the "data" dir.'
        self.assertIn(
            expected,
            os.listdir(os.path.join(BASE_DIR, 'data')),
            msg
        )

    def test_can_read_worksheet_names(self):
        """
        It should be able to read the worksheet names into a variable.
        """
        expected = 'NPU_October_Raw_Data'
        msg = '\n\nThe function did not read the worksheet from '\
              'the filesystem. Check if "data/worksheet_names.txt'\
              '" exists.'
        worksheet_names = helpers.read_worksheet_names()
        self.assertIn(expected, worksheet_names, msg)

    def test_at_least_one_worksheet_is_named_raw_data(self):
        """
        At least one of the variables in the worksheet list should 
        have some variation of "raw_data" in it.
        """
        expected = True
        msg = '\n\nNone of the lines in "data/worksheet_names.txt '\
              'contain a variation of the string "raw_data".'
        regex = re.compile(pattern=r'[\w\s\b]*(raw_data)[\w\s\b]*',
                           flags=re.IGNORECASE)
        names = helpers.read_worksheet_names()
        self.assertEqual(
            expected,
            any([regex.match(i) for i in names]),
            msg
        )

    def test_colums_to_keep_exists(self):
        """It should find the column names in the 'data' dir."""
        expected = 'columns_to_keep_clean.txt'
        msg = '\n\nThe function could not find the file named '\
              '"columns_to_keep_clean.txt" in the "data" dir.'
        self.assertIn(
            expected,
            os.listdir(os.path.join(BASE_DIR, 'data')),
            msg
        )

    def test_can_read_colums_to_keep(self):
        """
        It should be able to read the columns names into a variable.
        """
        expected = 'running_model'
        msg = '\n\nThe function did not read the columns from '\
              'the filesystem. Check if "data/columns_to_keep_clean.txt'\
              '" exists.'
        columns = helpers.read_columns_to_keep()
        self.assertIn(expected, columns, msg)

    def test_read_npu_data_reads_data(self):
        """
        The read_npu_data function should read the data as a pandas
        DataFrame.
        """
        expected = pd.DataFrame
        msg = '\n\nThe function could not read the "data/data.xlsx" '\
              'file.'
        df = helpers.read_npu_data(
            os.path.join(BASE_DIR, 'data', 'test_data', 'test_data.xlsx'),
            'NPU_October_Raw_Data',
        )
        self.assertIsInstance(df, expected, msg)

    def test_product_info_exists(self):
        """
        The read_product_info function should find the product
        information in the 'data' dir.
        """
        expected = 'product_info.csv'
        msg = '\n\nThe function could not find the file named '\
              '"product_info.csv" in the "data" dir.'
        self.assertIn(
            expected,
            os.listdir(os.path.join(BASE_DIR, 'data')),
            msg
        )

    def test_can_read_product_info(self):
        """
        The read_product_info function should return product
        information found in "data/product_info.csv".
        """
        expected = "running_model"
        msg = '\n\nThe function did not read the dataframe from '\
              'the filesystem. Check if "data/product_info.csv'\
              '" exists.'
        product_info = helpers.read_product_info()
        self.assertIn(expected, list(product_info.columns), msg)

    def test_product_info_is_read_into_a_pandas_dataframe(self):
        """
        The read_product_info function should return product
        information in a pandas dataframe. 
        """
        expected = pd.DataFrame
        msg = '\n\nThe function did not return a pandas DataFrame.'
        product_info = helpers.read_product_info()
        self.assertIsInstance(product_info, expected, msg)

    def test_rename_columns_slugifies(self):
        """
        The rename_columns function should slugify all column names.
        """
        expected = ['running_model', 'martha_stewart']
        msg = '\n\nThe function did not rename the columns properly'
        df = pd.DataFrame(columns=['RuNnInG=)(=+) M!,]odel', 'm$Ar%()tha-S$$te^wa*#rt'])
        self.assertEqual(
            expected,
            list(helpers.rename_columns(df).columns),
            msg
        )

    def test_rename_columns_returns_a_pandas_dataframe(self):
        """
        The rename_columns function should slugify all column names.
        """
        expected = pd.DataFrame
        msg = '\n\nThe rename_columns function did not return a '\
              'pandas DataFrame.'
        df = pd.DataFrame(columns=['RuNnInG=)(=+) M!,]odel', 'm$Ar%()tha-S$$te^wa*#rt'])
        self.assertIsInstance(df, expected, msg)


    def test_extract_product_info_returns_a_pandas_dataframe(self):
        """
        The extract_product_info function should return a pandas DataFrame.
        """
        expected = pd.DataFrame
        msg = '\n\nThe extract_product_info function did not return a '\
              'pandas DataFrame.'
        df = helpers.rename_columns(helpers.read_npu_data(
            os.path.join(BASE_DIR, 'data', 'test_data', 'test_data.xlsx'),
            'NPU_October_Raw_Data',
        ))
        self.assertIsInstance(
            helpers.extract_product_info(df),
            expected,
            msg
        )

    def test_extend_product_info_returns_a_pandas_dataframe(self):
        """
        The extend_product_info function should return a pandas
        dataframe.
        """
        expected = pd.DataFrame
        msg = '\n\nThe extend_product_info function did not return a '\
              'pandas DataFrame.'
        j = os.path.join
        data = os.path.join(BASE_DIR, 'data', 'test_data')
        p1 = pd.read_csv(j(data, 'test_product_info_1.csv'))
        p2 = pd.read_csv(j(data, 'test_product_info_2.csv'))
        self.assertIsInstance(
            helpers.extend_product_info(p1, p2),
            expected,
            msg
        )

    def test_extend_product_info_returns_unique_values(self):
        """
        The extend_product_info function should join unique values.
        """
        expected = None # expected reassigned to the actual value below
        msg = '\n\nThe extend_product_info function did not join '\
              'unique values'
        j = os.path.join
        data = os.path.join(BASE_DIR, 'data', 'test_data')
        p1 = pd.read_csv(j(data, 'test_product_info_1.csv'))
        p2 = pd.read_csv(j(data, 'test_product_info_2.csv'))
        expected = len(p1) + len(p2) - 1
        self.assertEqual(
            len(helpers.extend_product_info(p1, p2)),
            expected,
            msg
        )

    def test_extract_asc_info_returns_a_pandas_dataframe(self):
        """
        The extract_asc_info function should return a pandas DataFrame.
        """
        expected = pd.DataFrame
        msg = '\n\nThe extract_asc_info function did not return a '\
              'pandas DataFrame.'
        df = helpers.rename_columns(helpers.read_npu_data(
            os.path.join(BASE_DIR, 'data', 'test_data', 'test_data.xlsx'),
            'NPU_October_Raw_Data',
        ))
        self.assertIsInstance(
            helpers.extract_asc_info(df),
            expected,
            msg
        )

    def test_extend_asc_info_returns_a_pandas_dataframe(self):
        """
        The extend_asc_info function should return a pandas
        dataframe.
        """
        expected = pd.DataFrame
        msg = '\n\nThe extend_asc_info function did not return a '\
              'pandas DataFrame.'
        j = os.path.join
        data = os.path.join(BASE_DIR, 'data', 'test_data')
        p1 = pd.read_csv(j(data, 'test_asc_info_1.csv'))
        p2 = pd.read_csv(j(data, 'test_asc_info_2.csv'))
        self.assertIsInstance(
            helpers.extend_asc_info(p1, p2),
            expected,
            msg
        )

    def test_extend_asc_info_returns_unique_values(self):
        """
        The extend_asc_info function should join unique values.
        """
        expected = None # expected reassigned to the actual value below
        msg = '\n\nThe extend_asc_info function did not join '\
              'unique values'
        j = os.path.join
        data = os.path.join(BASE_DIR, 'data', 'test_data')
        p1 = pd.read_csv(j(data, 'test_asc_info_1.csv'))
        p2 = pd.read_csv(j(data, 'test_asc_info_2.csv'))
        expected = len(p1) + len(p2)
        self.assertEqual(
            len(helpers.extend_asc_info(p1, p2)),
            expected,
            msg
        )

    def test_extract_part_codes_info_returns_a_pandas_dataframe(self):
        """
        The extract_part_codes_info function should return a pandas DataFrame.
        """
        expected = pd.DataFrame
        msg = '\n\nThe extract_part_codes_info function did not return a '\
              'pandas DataFrame.'
        df = helpers.rename_columns(helpers.read_npu_data(
            os.path.join(BASE_DIR, 'data', 'test_data', 'test_data.xlsx'),
            'NPU_October_Raw_Data',
        ))
        self.assertIsInstance(
            helpers.extract_part_codes_info(df),
            expected,
            msg
        )

    def test_extend_part_codes_info_returns_a_pandas_dataframe(self):
        """
        The extend_part_codes_info function should return a pandas
        dataframe.
        """
        expected = pd.DataFrame
        msg = '\n\nThe extend_part_codes_info function did not return a '\
              'pandas DataFrame.'
        j = os.path.join
        data = os.path.join(BASE_DIR, 'data', 'test_data')
        p1 = pd.read_csv(j(data, 'test_part_codes_info_1.csv'))
        p2 = pd.read_csv(j(data, 'test_part_codes_info_2.csv'))
        self.assertIsInstance(
            helpers.extend_part_codes_info(p1, p2),
            expected,
            msg
        )

    def test_extend_part_codes_info_returns_unique_values(self):
        """
        The extend_part_codes_info function should join unique values.
        """
        expected = None # expected reassigned to the actual value below
        msg = '\n\nThe extend_part_codes_info function did not join '\
              'unique values'
        j = os.path.join
        data = os.path.join(BASE_DIR, 'data', 'test_data')
        p1 = pd.read_csv(j(data, 'test_part_codes_info_1.csv'))
        p2 = pd.read_csv(j(data, 'test_part_codes_info_2.csv'))
        expected = len(p1) + len(p2) - 1
        self.assertEqual(
            len(helpers.extend_part_codes_info(p1, p2)),
            expected,
            msg
        )

    def test_reduce_dataframe_removes_columns(self):
        """
        The reduce_dataframe function should keep only the columns
        described in data/columns_to_keep_clean.txt
        """
        expected = len(helpers.read_columns_to_keep())
        msg = '\n\nThe reduce_dataframe function did not remove'\
              ' unwanted columns.'
        data_dir = os.path.join(BASE_DIR, 'data', 'test_data')
        data = pd.read_excel(os.path.join(data_dir, 'test_data.xlsx'),
                           'NPU_October_Raw_Data')
        data = helpers.rename_columns(data)
        self.assertEqual(
            len(helpers.reduce_dataframe(data).columns),
            expected,
            msg
        )

    def test_reduce_dataframe_maintains_integrity(self):
        """
        The reduce_dataframe function remove duplicated columns while
        maintaining the integrity of the data (no rows should change).
        """
        expected = True
        msg = '\n\nThe reduce_dataframe function changed values at '\
              'certain indexes: [%s]'
        data_dir = os.path.join(BASE_DIR, 'data', 'test_data')
        data = pd.read_excel(os.path.join(data_dir, 'test_data.xlsx'),
                           'NPU_October_Raw_Data')
        cols = ['requested_date', 'running_model', 'asc_code']
        data = helpers.rename_columns(data).loc[:, cols]
        c_data = helpers.reduce_dataframe(data, cols[1:])
        integrity = []
        exceptions = []
        for (idx, row), (cidx, c_row) in zip(data.iterrows(), c_data.iterrows()):
            for col in c_data.columns.tolist():
                if row[col] == c_row[col]:
                    integrity.append(True)
                else:
                    integrity.append(False)
                    exceptions.append(str((row[col], c_row[col])))
        self.assertEqual(
            all(integrity),
            expected,
            msg % (';'.join([i for i in exceptions]))
        )

    def test_append_helper_columns_returns_a_pandas_dataframe(self):
        """
        The append_helper_columns function should return a pandas
        DataFrame.
        """
        expected = pd.DataFrame
        msg = '\n\nThe append_helper_columns function did not return '\
              'a pandas DataFrame.'
        data_dir = os.path.join(BASE_DIR, 'data', 'test_data')
        data = pd.read_excel(os.path.join(data_dir, 'test_data.xlsx'),
                           'NPU_October_Raw_Data')
        data = helpers.rename_columns(data)
        self.assertIsInstance(
            helpers.append_helper_columns(data),
            expected,
            msg,
        )

    def test_categorical_columns_exists(self):
        """
        It should find the categorical column names in the 'data' dir.
        """
        expected = 'categorical_columns.txt'
        msg = '\n\nThe function could not find the file named '\
              '"categorical_columns.txt" in the "data" dir.'
        self.assertIn(
            expected,
            os.listdir(os.path.join(BASE_DIR, 'data')),
            msg
        )

    def test_can_read_categorical_columns(self):
        """
        It should be able to read the categorical column names into a
        variable.
        """
        expected = 'service_type'
        msg = '\n\nThe function did not read the worksheet from '\
              'the filesystem. Check if "data/categorical_columns.txt'\
              '" exists.'
        column_names = helpers.read_categorical_columns()
        self.assertIn(expected, column_names, msg)

    def test_continuous_columns_exists(self):
        """
        It should find the continuous column names in the 'data' dir.
        """
        expected = 'continuous_columns.txt'
        msg = '\n\nThe function could not find the file named '\
              '"continuous_columns.txt" in the "data" dir.'
        self.assertIn(
            expected,
            os.listdir(os.path.join(BASE_DIR, 'data')),
            msg
        )

    def test_can_read_continuous_columns(self):
        """
        It should be able to read the continuous column names into a
        variable.
        """
        expected = 'labor_cost'
        msg = '\n\nThe function did not read the worksheet from '\
              'the filesystem. Check if "data/continuous_columns.txt'\
              '" exists.'
        column_names = helpers.read_continuous_columns()
        self.assertIn(expected, column_names, msg)


    def test_date_columns_exists(self):
        """
        It should find the date column names in the 'data' dir.
        """
        expected = 'date_columns.txt'
        msg = '\n\nThe function could not find the file named '\
              '"date_columns.txt" in the "data" dir.'
        self.assertIn(
            expected,
            os.listdir(os.path.join(BASE_DIR, 'data')),
            msg
        )

    def test_can_read_date_columns(self):
        """
        It should be able to read the date column names into a
        variable.
        """
        expected = 'bill_confirm_date'
        msg = '\n\nThe function did not read the worksheet from '\
              'the filesystem. Check if "data/date_columns.txt'\
              '" exists.'
        column_names = helpers.read_date_columns()
        self.assertIn(expected, column_names, msg)


    # def test_convert_categorical_columns(self):
    #     """
    #     The function convert_categorical_columns should return a
    #     pandas dataframe with correspondng date columns converted
    #     to pandas.DateTime64
    #     """
    #     expected = pd.CategoricalIndex
    #     msg = '\n\nThe function did not return columns typed as '\
    #           'pandas.DatetimeIndex.'
    #     data_dir = os.path.join(BASE_DIR, 'data', 'test_data')
    #     data = pd.read_excel(os.path.join(data_dir, 'test_data.xlsx'),
    #                        'NPU_October_Raw_Data')
    #     data = helpers.rename_columns(data)
    #     self.assertIsInstance(
    #         helpers.convert_categorical_columns(data),
    #         expected,
    #         msg,
    #     )


if __name__ == '__main__':
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(BASE_DIR)
    unittest.main()
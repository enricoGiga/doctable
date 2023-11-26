import unittest
from unittest import mock

from doctable.data_utility.datatype import Table
from doctable.table_recognition.recognition import TableRecognizer


class DoctableTestCase(unittest.TestCase):
    @mock.patch.object(TableDetector, 'get_cropped_tables', return_value=[Table((1, 4), (2, 7), cropped_image=np.ones(3))])
    @mock.patch.object(TableRecognizer, 'recognize', return_value=[{"res": "my mocked result"}])
    def test_table_detection_and_recognition(self, mock_recognize, mock_cropped_tables):
        image_path = "./data/page_test.JPG"
        # pdf_path = "/workspace/enrico/ailab-table-extraction/data/pdf/ATI+MORGANTI+SRL_LONG+TERM_2023_COO.pdf"
        total_res = table_extraction(image_path)
        mock_cropped_tables.assert_called_once()
        mock_recognize.assert_called_once()
        self.assertEquals(type(total_res), list)
        self.assertEquals(total_res[0].page_number, 1)
        self.assertEquals(total_res[0].page_number, 1)
        self.assertEquals(total_res[0].tables[0].recognition_results, "my mocked result")
        self.assertEquals(total_res[0].tables[0].pt1, (2, 7))
        self.assertEquals(total_res[0].tables[0].pt2, (1, 4))

if __name__ == '__main__':
    unittest.main()

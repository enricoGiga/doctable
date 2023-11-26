import settings
from doctable.table_recognition.recognition import TableRecognizer

if __name__ == '__main__':

    results = TableRecognizer().recognize("data/images/table.png")
    print(results)
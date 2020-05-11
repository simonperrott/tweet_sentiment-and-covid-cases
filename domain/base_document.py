from abc import ABC, abstractmethod
import csv
import os


class DocumentManager(ABC):

    def __init__(self, directory, filename, encoding, separator=',', headers=None):
        self.separator = separator
        self.directory = directory
        self.filename = filename
        self.headers = headers
        self.encoding = encoding

    @abstractmethod
    def create_document(self, file_row):
        """ transforms a domain object representing a document into our Document object """
        pass

    def get_qualified_file_name(self, filename):
        path = os.path.join('data', self.directory)
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, filename)
        return file_name

    def save_documents(self, mode, docs):
        file = self.get_qualified_file_name(self.filename)
        with open(file, mode) as csv_file:
            file_is_empty = os.stat(file).st_size == 0
            writer = csv.writer(csv_file, quotechar="\"", delimiter=',')
            try:
                if file_is_empty and self.headers:
                    writer.writerow(self.headers)
                writer.writerows(docs)
            except Exception as e:
                print(e)

    def load_documents(self):
        file = self.get_qualified_file_name(self.filename)
        if os.path.isfile(file):
            with open(file, encoding=self.encoding) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=self.separator, quotechar="\"")
                if self.headers is not None:
                    next(csv_reader)  # pass headers line
                documents = []
                for row in csv_reader:
                    documents.append(self.create_document(row))
            return documents
        return []
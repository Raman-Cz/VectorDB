class BM25Index:
    def __init__(self):
        self.documents = []

    def add_document(self, text):
        self.documents.append(text)

    def search(self, query):
        return [doc for doc in self.documents if query in doc]

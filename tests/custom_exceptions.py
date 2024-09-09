
class PythonMyException7(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        s = "[PythonMyException7]: " + self.message.a
        return s

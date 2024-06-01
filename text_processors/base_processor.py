from abc import ABC


class BaseProcessor(ABC):

    def get_tokens(self):
        return self.tokens

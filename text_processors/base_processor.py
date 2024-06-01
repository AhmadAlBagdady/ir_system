from abc import ABC


class BaseProcessor(ABC):
    tokens:str
    inputs:str|list

    def get_tokens(self):
        return self.tokens

    def to_string(self) -> str:
        return str.join(' ',self.tokens)
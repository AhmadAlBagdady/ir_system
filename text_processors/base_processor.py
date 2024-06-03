from abc import ABC


class BaseProcessor(ABC):
    inputs:str|list
    
    tokens:str

    def get_tokens(self):
        return self.tokens

    def to_string(self) -> str:
        return str.join(' ',self.tokens)
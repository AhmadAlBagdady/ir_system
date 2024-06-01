import re

text = "asdfsdafhioi https://asdfsf.com asdfasdf"

def remove_urls(text):
    url_pattern = r'https?://\S+|www\.\S+'
    clean_text = re.sub(url_pattern, '', text)
    return clean_text

print(remove_urls(text))
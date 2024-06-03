from text_processors.remove_numbers_processor import RemoveNumbersProcessor
from text_processors.remove_urls_processor import RemoveUrlsProcessor

remove = RemoveNumbersProcessor('fuck numbers like 1, 2 and even 73')

removeUrls = RemoveUrlsProcessor("fuck you omar http://asdfh.com , https://asdfasd.com asdfasdf")
print(removeUrls.tokens)
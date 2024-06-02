from text_processors.remove_numbers_processor import RemoveNumbersProcessor

remove = RemoveNumbersProcessor('fuck numbers like 1, 2 and even 73')

print(remove.tokens)
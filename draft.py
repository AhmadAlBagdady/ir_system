def dd(text):
    print(text)
    quit()


csv_data = load_csv('/home/baraa/Desktop/wikIR1k/documents.csv')

index = build_inverted_index()

tf_idf_df = calculate_tf_idf()

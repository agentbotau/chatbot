from info_retreival import load_rtf_files_in_batch, create_and_save_vectorstore

docs=load_rtf_files_in_batch('C:/Users/HP/Downloads/Part 1-2')
create_and_save_vectorstore(docs)

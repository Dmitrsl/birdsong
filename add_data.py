import zipfile
import os


zip_files = ['726312_1342586_bundle_archive.zip', '726237_1342541_bundle_archive.zip']
directory_to_extract_to = 'new_data'

for zip_file in zip_files:
    with zipfile.ZipFile(f'data/{zip_file}', 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

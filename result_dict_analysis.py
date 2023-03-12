from glob import glob
import os
import json

list_of_result_dicts_files = glob("./**/*result_dict.txt", recursive=True)


for result_dict_filename in list_of_result_dicts_files:
    print(f"Analyzing {result_dict_filename}")
    result_dict_directory = os.path.dirname(os.path.abspath(result_dict_filename))
    result_dict_file = open(result_dict_filename, "r")
    analyzed_dict_file = open(result_dict_filename[:-4]+ "_analyzed.txt", "w")
    imported_dictionary = json.load(result_dict_file)
    for key in imported_dictionary:
        analyzed_dict_file.write(f"average_{key} : {sum(imported_dictionary[key])/len(imported_dictionary[key])}\n")
    analyzed_dict_file.close()
import sys
import glob
import os

validated_datasets = ["MAPS", "GuitarSet"]
number_of_compared_models = len(sys.argv) - 1
models_to_be_compared = sys.argv[1:]


def get_result_dict(filename):
    result_dict = {}
    with open(filename, "r") as opened_file:
        for elem in opened_file.readlines():
            name = elem.split(":")[0].strip()
            value = float(elem.split(":")[1])
            result_dict[name] = value
    return result_dict

class BoldResolver:
    def __init__(self, result_dict):
        self.result_dict = result_dict
        self.key_order = {}

            



class ResultsAggregator:
    def __init__(self, results_dict):
        self.result_dict = results_dict
        self.key_order = {}
        self.maximum_indexes_dict = {}

        for key in self.result_dict:
            for tested_dataset_idx in range(len(self.result_dict[key])):
                self.analyze_properties(tested_dataset_idx)

    def analyze_properties(self, tested_dataset_index):
            self.maximum_indexes_dict[f"note_precision_{tested_dataset_index}"] = self.__analyze_property("note_precision", tested_dataset_index)
            self.maximum_indexes_dict[f"note_recall_{tested_dataset_index}"] = self.__analyze_property("note_recall", tested_dataset_index)
            self.maximum_indexes_dict[f"note_F1_{tested_dataset_index}"] = self.__analyze_property("note_F1", tested_dataset_index)
            self.maximum_indexes_dict[f"frame_precision_{tested_dataset_index}"] = self.__analyze_property("frame_precision", tested_dataset_index)
            self.maximum_indexes_dict[f"frame_recall_{tested_dataset_index}"] = self.__analyze_property("frame_recall", tested_dataset_index)
            self.maximum_indexes_dict[f"frame_F1_{tested_dataset_index}"] = self.__analyze_property("frame_F1", tested_dataset_index)

    def __analyze_property(self, property_name, tested_dataset_index):
        list_of_properties = []
        for i, key in enumerate(self.result_dict):
            self.key_order[key] = i
            list_of_properties.append(getattr(self.result_dict[key][tested_dataset_index], property_name))
        return list_of_properties.index(max(list_of_properties))
    
    def get_line_of_note_results(self, key, tested_dataset_index):
        validated_dataset_result = self.result_dict[key]
        note_attributes = ["note_precision", "note_recall", "note_F1"]
        line = ""
        for attribute in note_attributes:
            if self.key_order[key] == self.maximum_indexes_dict[f"{attribute}_{tested_dataset_index}"]:
                line += "& \\textbf{"
                line += "{:.3f}".format(getattr(validated_dataset_result[tested_dataset_index], attribute))
                line += "} "
            else:
                line += "& {:.3f} ".format(getattr(validated_dataset_result[tested_dataset_index], attribute))
        return line

    def get_line_of_frame_results(self, key, tested_dataset_index):
        validated_dataset_result = self.result_dict[key]
        note_attributes = ["frame_precision", "frame_recall", "frame_F1"]
        line = ""
        for attribute in note_attributes:
            if self.key_order[key] == self.maximum_indexes_dict[f"{attribute}_{tested_dataset_index}"]:
                line += "& \\textbf{"
                line += "{:.3f}".format(getattr(validated_dataset_result[tested_dataset_index], attribute))
                line += "} "
            else:
                line += "& {:.3f} ".format(getattr(validated_dataset_result[tested_dataset_index], attribute))
        return line

    def get_line_of_latex_table(self, key, tested_dataset_index):
        validated_dataset_result = self.result_dict[key]
        note_attributes = ["note_precision", "note_recall", "note_F1", "frame_precision", "frame_recall", "frame_F1"]
        line = ""
        for attribute in note_attributes:
            if self.key_order[key] == self.maximum_indexes_dict[f"{attribute}_{tested_dataset_index}"]:
                line += "& \\textbf{"
                line += "{:.3f}".format(getattr(validated_dataset_result[tested_dataset_index], attribute))
                line += "} "
            else:
                line += "& {:.3f} ".format(getattr(validated_dataset_result[tested_dataset_index], attribute))
        return line

class ValidatedDatasetResult:
    def __init__(self):
        self.note_precision = None
        self.note_recall = None
        self.note_F1 = None
        self.frame_precision = None
        self.frame_recall = None
        self.frame_F1 = None

    def setResults(self, note_precision, note_recall, note_F1, frame_precision, frame_recall, frame_F1):
        self.note_precision = note_precision
        self.note_recall = note_recall
        self.note_F1 = note_F1
        self.frame_precision = frame_precision
        self.frame_recall = frame_recall
        self.frame_F1 = frame_F1
    
    def printAllResults(self):
        print(f"note precision: {self.note_precision}\
                note recall: {self.note_recall}\
                note F1: {self.note_F1}\
                frame precision: {self.frame_precision}\
                frame recall: {self.frame_recall}\
                frame F1: {self.frame_F1}\
                ")

    def loadFromFile(self, filename):
        print(f"Loading result from file: {filename}")
        result_dict = get_result_dict(filename)
        self.note_precision = result_dict["average_metric/note/precision"]
        self.note_recall = result_dict["average_metric/note/recall"]
        self.note_F1 = result_dict["average_metric/note/f1"]
        self.frame_precision = result_dict["average_metric/frame/precision"]
        self.frame_recall = result_dict["average_metric/frame/recall"]
        self.frame_F1 = result_dict["average_metric/frame/f1"]
    
    def get_line_of_latex_table(self):
        return "& {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(self.note_precision,
                                                                            self.note_recall,
                                                                            self.note_F1,
                                                                            self.frame_precision,
                                                                            self.frame_recall,
                                                                            self.frame_F1)
    def get_line_of_note_results(self):
        return "& {:.3f} & {:.3f} & {:.3f}".format(self.note_precision,
                                                   self.note_recall,
                                                   self.note_F1)

    def get_line_of_frame_results(self):
        return "& {:.3f} & {:.3f} & {:.3f}".format(self.frame_precision,
                                                   self.frame_recall,
                                                   self.frame_F1)




print(f"Number of compared models: {number_of_compared_models} Models to be compared: {models_to_be_compared}")
results_dictionary = {}
for i in range(number_of_compared_models):
    model_directory = models_to_be_compared[i]
    print(f"Adding '{model_directory}' row to analysis")
    results_dictionary[model_directory] = []
    for validated_dataset in validated_datasets:
        seatch_pattern = f"*eval/VALIDATION_ON_{validated_dataset}*"
        directory_containing_validation = glob.glob(f"{model_directory}/*{seatch_pattern}")[0]
        validation_result_file = os.path.join(directory_containing_validation, f"{validated_dataset}_result_dict_analyzed.txt")
        validated_dataset_result = ValidatedDatasetResult()
        validated_dataset_result.loadFromFile(validation_result_file)
        validated_dataset_result.printAllResults()
        results_dictionary[model_directory].append(validated_dataset_result)
        print(validated_dataset_result.get_line_of_latex_table())

def newline():
    print("    \hline")

def endline():
    print("    \\\\")

class LatexTableGenerator:
    def __init__(self, results_dictionary, validated_datasets):
        self.result_dictionary = results_dictionary
        self.validated_datasets = validated_datasets
       
        for key in results_dictionary:
            assert len(results_dictionary[key]) == len(self.validated_datasets)

    def generate_table(self, caption, label="results", note=True, frame=True):
        assert note or frame
        results_aggregator = ResultsAggregator(self.result_dictionary)
        header = ""
        if note and frame:
            header += "\\begin{landscape}\n"
        header += "\\begin{table}[]\n"
        header += "    \\centering\n"
        header += "    \\caption{"
        header += caption
        header += "}\n"

        if note and frame:
            columns_configuration = "{|l"+"|c"*6*len(validated_datasets)+"|}"
        else:
            columns_configuration = "{|l"+"|c"*3*len(validated_datasets)+"|}"
        tabular_start = "    \\begin{tabular}" + columns_configuration
        print(header)
        print(tabular_start)

        newline()
        print("    Testing dataset", end=" ")
        for dataset in self.validated_datasets:
            dataset_name = "{\\textsc{" + f"{dataset}" + "}}"
            if note and frame:
                print("& \multicolumn{6}{|c|}"+dataset_name, end=" ")
            else:
                print("& \multicolumn{3}{|c|}"+dataset_name, end=" ")
        endline()

        newline()
        print("     ", end="")
        for dataset in self.validated_datasets:
            columns = ""
            if note:
                columns += "& \multicolumn{3}{|c|}{note} "
            if frame:
                columns += "& \multicolumn{3}{|c|}{frame} "
            print(columns, end = " ")
        endline()

        newline()
        print("     ", end="")
        for dataset in self.validated_datasets:
            if note and frame:
                print("& P & R & F1 & P & R & F1 ", end=" ")
            else:
                print("& P & R & F1", end=" ")
        endline()

        for key in self.result_dictionary:
            newline()
            trained_model_name = key.split("_")[-1]
            print(f"    {trained_model_name}", end=" ")
            tested_datasets = self.result_dictionary[key]
            for tested_dataset_idx in range(len(tested_datasets)):
                if note:
                    print(results_aggregator.get_line_of_note_results(key, tested_dataset_idx), end=" ")
                if frame:
                    print(results_aggregator.get_line_of_frame_results(key, tested_dataset_idx), end=" ")
            endline()

        ending = "    \\end{tabular}\n"
        ending += "    \\label{tab:"
        ending += label
        ending += "}\n"
        ending += "\\end{table}"
        if note and frame:
            ending += "\\end{landscape}\n"
        newline()
        print(ending)
        # for dataset in self.validated_datasets
            


latex_table_generator = LatexTableGenerator(results_dictionary, validated_datasets)
latex_table_generator.generate_table(caption="All results", label="all_results")

latex_table_generator.generate_table(caption="note results",label="note_results", note=True, frame=False)
latex_table_generator.generate_table(caption="frame results",label="frame_results", note=False, frame=True)

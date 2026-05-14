##############################
# The configurator lives in  #
# that extra module to avoid #
# the circular import issue  #
##############################

from prettyconsole import Console
from tools import DictionaryTools
import os, sys, json


class Configurator:
    """
    For loading paths and configurations from a json file. If necessary, different
    instances for different config files can be created.
    """

    def __init__(self, path_folder: str, file_name: str, info: str = None) -> None:
        if os.path.exists(path_folder):  # check if the path to the folder exists
            self.path_folder: str = path_folder  # path to the config file
        else:
            Console.printf("error", f"Path does not exists: {path_folder}. Terminate program!")
            sys.exit()

        self.file_name: str = file_name  # file name or desired file name
        self.file_path: str = os.path.join(self.path_folder, self.file_name)  # file name and path
        self.info: str = info  # additional information of the configurator instance
        self.data: dict = None

    def load(self) -> object:
        """
        For loading a json file and storing it as a dictionary.

        :return: Nothing
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.data = json.load(file)
        else:
            Console.printf("error", f"Could not load '{self.file_name}'. Wrong path, name or may not exist!")

        return self

    def save(self, new_data: dict):
        """
        For creating or overwriting a formatted json config file from a dictionary.

        :param new_data:
        :return:
        """

        if os.path.exists(self.file_path):
            Console.ask_user(f"Overwrite file '{self.file_path}' ?")
        else:
            Console.printf("info", f"New file '{self.file_name}' will be created")

        try:
            with open(self.file_path, "w") as file:
                json.dump(new_data, file, indent=4, default=str)  # default str converts everything not known to a string
        except Exception as e:
            Console.printf("error", f"Could not create/overwrite the file: {type(e).__name__}: {e}")

    def print_formatted(self) -> None:
        """
        For printing the content of the JSON file to the console.

        :return: None
        """

        Console.printf("info", f"Content of the config file: {self.file_name} \n"
                               f"{json.dumps(self.data, indent=4)}")


    def get_data(self, key: str | list):
        """
        To get data out of the dict, which represents the loaded jason file.

        Possible to access data via (examples):
            data(key="metabolites")
            data(key="txt.metabolites.path")
            data(key=["txt.metabolites", "path"])
            ... and so on.

        :param key: to get the (possible also nested) data from the loaded in the dict
        :return: dict, values, ...
        """

        return DictionaryTools.get(self.data, key)
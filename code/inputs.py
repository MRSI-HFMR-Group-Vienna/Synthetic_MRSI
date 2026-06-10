##############################
# The configurator lives in  #
# that extra module to avoid #
# the circular import issue  #
##############################

from prettyconsole import Console
from tools import DictionaryTools, PathTools
import os, sys, json
from interface import ResourceInterface

class JsonResources(ResourceInterface):
    """
    This is the base class for loading data from JSON files.
    """
    def __init__(self, path_folder: str, file_name: str, info: str = None) -> None:
        if not os.path.exists(path_folder):
            Console.printf("error", f"Path does not exists: {path_folder}. Terminate program!", raise_error=FileNotFoundError)

        self.path_folder: str = path_folder
        self.file_name: str = file_name
        self.file_path: str = os.path.join(self.path_folder, self.file_name)
        self.info: str = info
        self.data: dict = None

    def load(self) -> "JsonResources":
        """
        For loading a json file and storing it as a dictionary.

        :return: The object of this class.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.data = json.load(file)
        else:
            Console.printf("error", f"Could not load '{self.file_name}'. Wrong path, name or may not exist!", raise_error=FileNotFoundError)

        return self

    def save(self, new_data: dict) -> None:
        """
        To overwrite the old content of the json file.

        :param new_data: The modified dict that should replace the content in the JSON file.
        :return: Nothing
        """
        if os.path.exists(self.file_path):
            Console.ask_user(f"Overwrite file '{self.file_path}' ?")
        else:
            Console.printf("info", f"New file '{self.file_name}' will be created")

        try:
            with open(self.file_path, "w") as file:
                json.dump(new_data, file, indent=4, default=str)
        except Exception as e:
            Console.printf("error", f"Could not save/overwrite the file: {type(e).__name__}: {e}")

    def print_formatted(self) -> None:
        """
        To print the content of dict gained from the JSON file formatted to the console.
        :return:
        """
        Console.printf("info", f"Content of the config file: {self.file_name} \n")

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


class Configurator(JsonResources):
    """
    For loading paths and configurations from a json file. If necessary, different
    instances for different config files can be created.

    (!) Note: At the moment has the same functionalities then the JsonResources basis class,
              but it can be extended with further methods.
    """

class MetaboliteLibrary(JsonResources):
    """
    For loading a JSON file that holds various information about the metabolites, water,
    macromolecules, ...)

    (!) Note: Only has functionality of the basis class at the moment, can be extended with
              new methods in the future.
    """

class SimulationParameters(JsonResources):
    """
    For loading a JSON file that holds various information about the simulation parameters,

    (!) Note: Only has functionality of the basis class at the moment, can be extended with
              new methods in the future.
    """

class CitationManager:
    pass
    # TODO: Not yet implemented ==> at the moment in tools ==> move it from tools.CitationManager to resources.CitationManager



# TODO: Add citation manager (name class Bibliography)
# TODO: Add MetaboliteLibrary (name the class so, is for getting data from chemical_compounds_OLD.json)
# TODO: Add SimulationParameters (need to create .json file for this, or other, which might be more human readable?)
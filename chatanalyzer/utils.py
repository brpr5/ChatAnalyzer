
# Standard library imports


# Third party imports

from colorama import init, Fore, Back
init(autoreset=True)

# Local application imports

class constants:

    logger_name = "chatanalyzerLogger"
    #TODO: What should the better practive here?


def print_successful(msg, success=True):
    """ Prints using colorama if message is succesful or not

    Args:
        msg (str): message that I want to print
        success (bool, optional): if True, color Green else Red. Defaults to True.
    """

    color = Fore.GREEN if success else Fore.RED

    print(color + msg)

if __name__ == "__main__":
    print(constants.logger_name)
    print_successful(__name__, True)
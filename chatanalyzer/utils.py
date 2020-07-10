
# Standard library imports


# Third party imports

from colorama import init, Fore, Back
init(autoreset=True)

# Local application imports


def print_successful(msg, success=True):
    """ Prints using colorama if message is succesful or not

    Args:
        msg (str): message that I want to print
        success (bool, optional): if True, color Green else Red. Defaults to True.
    """

    color = Fore.GREEN if success else Fore.RED

    print(color + msg)
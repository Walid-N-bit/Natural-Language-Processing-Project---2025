"""
a class to colorize console output text.

example:
print(bcolors.RED + "this is a red text." + bcolors.ENDC)
or:
print(f"{bcolors.GREEN}this is a green text. {bcolors.ENDC}")
"""


class bcolors:
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

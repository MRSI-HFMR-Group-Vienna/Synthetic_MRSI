import default

from colorama import Fore, Back, Style
import textwrap
import time
import sys

class Console:
    __counter = 1  # Global __counter which will increase with each call of the printf function.
    __max_message_length = 100  # Space in the console reserved for the text of the user
    __indent = " " * 22  # Intent for the long text
    __time_previous = 0
    __collected_lines = ""

    @staticmethod
    def printf(status: str, message: str, long_format: bool = False, long_annotation: str = "", mute: bool = False) -> None:
        """
        A formatted output. The user can insert the status and the corresponding message.

        :param status: possible options as strings: success, error, warning.
        :param message: a desired string message with length <= 100 or a message which contains several lines, where each
                        line is <= 100 and ends with an \n
        :param long_format: default is False. If true a special case, where several lines are printed and "--v" is displayed.
        :param long_annotation: default is "", thus empty string. Just use it if long_format is set to True. Then, an
                                annotation in form of (my annotation) is created after the arrow pointing to the text.
                                Thus: [ i ][ message ] ---v  (my annotation)
        :param mute: If true then no console output. Maybe useful for already implemented printf. Thus, no need to comment.
        :return: None
        """
        if mute: return

        # Colors for the respective status
        colors = {
            'success': Fore.LIGHTGREEN_EX,
            'error': Fore.LIGHTRED_EX,
            'warning': Fore.LIGHTYELLOW_EX,
            'info': Fore.LIGHTBLUE_EX,
        }

        long_annotation = f" ({long_annotation})" if long_annotation != "" else ""

        # Distinguish between a "normal" message length and a "long" one. Thus, print the long on in the next line.
        if (len(message) <= Console.__max_message_length) and (long_format is False):
            output = f"[{Console.__counter:^5}][{colors[status] + status + Style.RESET_ALL:^18}] >> {message:<100} \n"
        else:
            output = f"[{Console.__counter:^5}][{colors[status] + status + Style.RESET_ALL:^18}] ---v {long_annotation}"

            # Distinguish between a long text already formatted by including "\n" and not formatted long text. If the
            # text is not formatted and long then an automatically new lines are generated. Further, the text will be
            # indented.
            if "\n" in message:  # for keeping the formatted output
                lines = message.split('\n')
                lines[0] = "\n" + Console.__indent + lines[0]
            else:  # formation by automatic line break
                output += " (automatic line breaks) \n"
                lines = textwrap.fill(message, width=Console.__max_message_length).split('\n')

            # Just print the long text line by line
            for line in lines:
                output += Console.__indent + line + "\n"

        print(output, end="")

        Console.__counter += 1  # Just increasing the global counter.

    @staticmethod
    def add_lines(line: str) -> None:
        """
        Add lines which can then be printed collected with @print_collected_lines

        :param line: Some string.
        :return: Nothing
        """
        Console.__collected_lines += line + "\n"

    @staticmethod
    def printf_collected_lines(status: str, mute: bool = False) -> None:
        """
        Print the collected lines by the method @add_lines

        :param status: according to the statuses defined in @printf
        :param mute: mutes the output.
        :return: Nothing
        """
        Console.printf(status, Console.__collected_lines, long_format=True, long_annotation="collected several lines", mute=mute)
        Console.__collected_lines = ""

    @staticmethod
    def printf_section(title: str) -> None:
        """
        For creating a new section in the console. The background is set to white.

        :param title: Desired title of the section.
        :return: Nothing
        """
        print()
        title = f"SECTION: {title:100}"
        print(f"{Back.WHITE + title + Style.RESET_ALL}")

    @staticmethod
    def ask_user(message: str, exit_if_false: bool = True) -> bool:
        """
        To ask the user to continue or terminate the program. Another option is to return either True or False.
        Example usage: if the required estimated space exceeds the desired limit.

        :return:
        """
        answer = input(f"{Back.LIGHTYELLOW_EX + Fore.BLACK + '[CONTINUE (y/n) ?]' + Style.RESET_ALL + ' >> '}{message} -> ").lower()
        if answer == "n":

            if exit_if_false is True:
                Console.printf("error", "The user has terminated the program!")
                sys.exit()
            else:
                return False

        return True

    @staticmethod
    def check_condition(logic_operation: bool, ask_continue: bool = False) -> None:
        """
        For simply check boolean operations and if they are not true then ask the user whether to continue the program. An example could be
        memory_used < 5 GB.

        :param logic_operation: only boolean operations allowed -> Thus, True or False as result.
        :param ask_continue: If True, the user will be asked to continue the program if the boolean operation is False or to exit. By default,
                            False, thus if boolean operation results in False, the program will be terminated automatically.
        :return: Nothing
        """
        if logic_operation:
            Console.printf("success", f"Logic operation {logic_operation}. Continue program.")
        else:
            if not ask_continue:
                Console.printf("error", f"Logic operation {logic_operation}. Terminate program.")
                sys.exit()
            else:
                answer = input(f"{Back.LIGHTCYAN_EX + Fore.BLACK + '[CONTINUE (y/n) ?] >> ' + Style.RESET_ALL} Logic operation {logic_operation}").lower()
                if answer:
                    Console.printf("info", "The user has terminated the program!")
                else:
                    Console.printf("error", f"Logic operation {logic_operation}. Terminate program.")
                    sys.exit()

    @staticmethod
    def reset_counter() -> None:
        """
        Reset the global __counter to 1. A message will be printed to the console.

        :return: None
        """
        print(f"{Back.WHITE}{'RESET counter TO 1':^30}{Style.RESET_ALL}")
        # global __counter
        Console.__counter = 1

    @staticmethod
    def start_timer() -> None:
        """
        Start the timer and stop it with @stop_timer

        :return: Nothing
        """
        print(f"{Back.LIGHTBLUE_EX + Fore.BLACK}{'START TIMER':^30}{Style.RESET_ALL}")
        Console.__time_previous = time.time()

    @staticmethod
    def stop_timer() -> None:
        """
        Stops the timer started with @start_timer and prints the time passed to the console.

        :return: Nothing
        """
        took_time = f"TOOK {round(time.time() - Console.__time_previous, 3)} sec"
        print(f"{Back.LIGHTBLUE_EX + Fore.BLACK}{took_time:^30}{Style.RESET_ALL}")


# Just a few examples
if __name__ == "__main__":
    '''
    # Example usage:
    Console.printf('success', 'Operation completed successfully.')
    Console.start_timer()
    Console.reset_counter()
    Console.printf('error', 'An error occurred while processing data.')
    time.sleep(1)
    Console.stop_timer()
    # reset___counter()
    Console.printf('warning', 'Warning: Incomplete data detected.')
    Console.printf('success',
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.')
    Console.printf('success',
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.\n'
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.\n'
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.')
    Console.printf('success',
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.')
    Console.printf('success',
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.')
    Console.printf('error',
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.\n'
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.\n'
                   'This is a longer output message that needs indentation. It will span multiple lines and should be properly indented.')
    Console.printf('info',
                   'This is a longer output message that needs indentation. It will not span multiple lines and should be properly indented.')

    Console.add_lines(f"A {1}")
    Console.add_lines("B")
    Console.printf_collected_lines("info")
    # Console.printf(message="Test")

    # Console.start_timer()
    # time.sleep(1)
    # Console.stop_timer()
    '''

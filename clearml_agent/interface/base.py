from __future__ import print_function

import abc
import argparse
from copy import deepcopy
from functools import partial

from .._vendor import six
from .._vendor.pathlib2 import Path

from clearml_agent import definitions
from clearml_agent.session import Session

HEADER = 'CLEARML-AGENT Deep Learning DevOps'


class Parser(argparse.ArgumentParser):
    __default_subparser = None

    def __init__(self, usage_on_error=True, *args, **kwargs):
        super(Parser, self).__init__(fromfile_prefix_chars=definitions.FROM_FILE_PREFIX_CHARS, *args, **kwargs)
        self._usage_on_error = usage_on_error

    @property
    def choices(self):
        try:
            subparser = next(
                action for action in self._actions
                if isinstance(action, argparse._SubParsersAction))
        except StopIteration:
            return {}
        return subparser.choices

    def error(self, message):
        if self._usage_on_error and message == argparse._('too few arguments'):
            self.print_help()
            print()
            self.exit(2, argparse._('%s: error: %s\n') % (self.prog, message))
        super(Parser, self).error(message)

    def __getitem__(self, name):
        return self.choices[name]

    def remove_top_level_results(self, parse_results):
        """
        Remove useless, artifact values
        :param parse_results: resulting namespace of parse_args, converted to dict ( vars(args) )
        """
        for action in self._actions:
            if action.dest != 'version':
                parse_results.pop(action.dest, None)
        for key in ('func', 'command', 'subcommand', 'action'):
            parse_results.pop(key, None)

    def set_default_subparser(self, name):
        self.__default_subparser = name

    def get_default_subparser(self):
        return self.choices[self.__default_subparser]

    def _parse_known_args(self, arg_strings, *args, **kwargs):
        in_args = set(arg_strings)
        d_sp = self.__default_subparser
        if d_sp is not None and not {'-h', '--help'}.intersection(in_args):
            for x in self._subparsers._actions:
                subparser_found = (
                        isinstance(x, argparse._SubParsersAction) and
                        in_args.intersection(x._name_parser_map.keys())
                )
                if subparser_found:
                    break
            else:
                # insert default in first position, this implies no
                # global options without a sub_parsers specified
                arg_strings = [d_sp] + arg_strings
        return super(Parser, self)._parse_known_args(
            arg_strings, *args, **kwargs
        )


class AliasedPseudoAction(argparse.Action):
    """
    Action for choosing between sub-commands, including aliases
    """
    def __init__(self, name, aliases, help):
        dest = name
        aliases = [a for a in aliases if a != name]
        if aliases:
            dest += ' (%s)' % ','.join(aliases)
        super(AliasedPseudoAction, self).__init__(option_strings=[], dest=dest, help=help)


class AliasedSubParsersAction(argparse._SubParsersAction):
    """
    Action for adding aliases for sub-commands
    """

    def add_parser(self, name, **kwargs):
        aliases = kwargs.pop('aliases', [])
        parser = super(AliasedSubParsersAction, self).add_parser(name, **kwargs)

        # Make the aliases work
        for alias in aliases:
            self._name_parser_map[alias] = parser
        # Make the help text reflect them, first removing old help entry.
        help = kwargs.pop('help', None)
        if help:
            self._choices_actions.pop()
            pseudo_action = AliasedPseudoAction(name, aliases, help)
            self._choices_actions.append(pseudo_action)

        return parser


class OnlyPluralChoicesHelpFormatter(argparse.HelpFormatter):

    @staticmethod
    def _metavar_formatter(action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choice_strs = [str(choice) for choice in action.choices]
            choice_strs = [choice for choice in choice_strs if choice + 's' not in choice_strs]
            result = '{%s}' % ','.join(choice_strs)
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format


def hyphenate(s):
    return s.replace('_', '-')


def add_args(parser, args):
    """
    Add arguments to parser from args mapping
    :param parser: parser to add arguments to
    :type parser: argparse.ArgumentParser
    :param args: mapping of name -> other arguments to ArgumentParser.add_argument
    :type args: dict
    """
    for arg_name, arg_params in args.items():
        aliases = arg_params.pop('aliases', tuple())
        parser.add_argument(arg_name, *aliases, **arg_params)


def add_mutually_exclusive_groups(parser, groups):
    """
    Add mutually exclusive groups to parser from list
    :param parser: parser to add groups to
    :param groups: list of dictionaries, each containing:
                   1. 'args': parameter to add_args
                   2. arguments to ArgumentParser.add_mutually_exclusive_group
    """
    for group in groups:
        args = group.pop('args', {})
        group_parser = parser.add_mutually_exclusive_group(**group)
        add_args(group_parser, args)


def add_service(subparsers, name, commands, command_name_dest='command', formatter_class=argparse.RawDescriptionHelpFormatter, **kwargs):
    """
    Add service commands to parser from arguments dictionary
    :param subparsers: subparsers object of ArgumentParser
    :param name: name of service
    :param commands: mapping of names to dictionaries, each of them containing:
                     1. 'args' - mapping of name -> other arguments to ArgumentParser.add_argument
                     2. 'help' - command description
                     3. 'mutually_exclusive_groups' - see add_mutually_exclusive_groups
    :param command_name_dest: name of attribute in which to store selected sub-command
    :param formatter_class; help formatter class
    :param kwargs: any other arguments to add_parser method of subparser object
    :return: service subparser
    """
    commands = deepcopy(commands)
    service_parser = subparsers.add_parser(
        name,
        # aliases=(name.strip('s'),),
        formatter_class=formatter_class,
        **kwargs
    )
    service_parser.register('action', 'parsers', AliasedSubParsersAction)
    service_parser.set_defaults(**{command_name_dest: name})
    service_subparsers = service_parser.add_subparsers(
        title='{} commands'.format(name.capitalize()),
        parser_class=partial(Parser, usage_on_error=False),
        dest='action')

    # This is a fix for a bug in python3's argparse: running "clearml-agent some_service" fails
    service_subparsers.required = True

    for name, subparser in commands.pop('subparsers', {}).items():
        add_service(service_subparsers, name, command_name_dest='subcommand', **subparser)

    for command_name, command in commands.items():
        command_type = command.pop('type', None)
        mutually_exclusive_groups = command.pop('mutually_exclusive_groups', [])
        func = command.pop('func', command_name)
        args = command.pop('args', {})
        command_parser = service_subparsers.add_parser(hyphenate(command_name), **command)
        if command_type:
            command_type.make(command_parser)
        command_parser.set_defaults(func=func)
        add_mutually_exclusive_groups(command_parser, mutually_exclusive_groups)
        add_args(command_parser, args)

    return service_parser


@six.add_metaclass(abc.ABCMeta)
class CommandType(object):

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def make(self, parser):
        return self._make(parser, *self._args, **self._kwargs)

    @abc.abstractmethod
    def _make(self, *args, **kwargs):
        pass


class ListCommand(CommandType):

    @staticmethod
    def _make(parser, default_value, pagination=False, tree=False):
        if tree:
            table_group = parser.add_mutually_exclusive_group()
            tree_action = table_group.add_argument(
                '--tree', help='Tree view output', action='store_true', default=False)
            csv_group = parser.add_mutually_exclusive_group()
            # hack: tree cannot be used with either csv or table, which can be used
            # which each other
            csv_group._group_actions.append(tree_action)

        else:
            csv_group = table_group = parser

        table_group.add_argument(
            '--table',
            help='Select table columns ("#" separated, default: %(default)s)',
            default=default_value)
        csv_group.add_argument(
            '--csv',
            help='Generate CSV output to specified path',
            default=None)
        parser.add_argument(
            '--no-headers',
            action='store_false',
            dest='headers',
            help='Do not print table/csv headers')
        parser.add_argument(
            '--sort',
            help='Fields to sort by (same format as --table)')
        parser.add_argument(
            '--ascending',
            default=None,
            dest='sort_reverse',
            action='store_false',
            help='Sort in ascending order (default)')
        parser.add_argument(
            '--descending',
            default=None,
            dest='sort_reverse',
            action='store_true',
            help='Sort in descending order')

        if not pagination:
            return

        parser.add_argument(
            '--page',
            help='Page number to show (default: %(default)s)',
            default=0,
            type=bound_number_type(minimum=0))
        parser.add_argument(
            '--page-size',
            help='Size of page (default: %(default)s)',
            default=50,
            type=bound_number_type(minimum=1))
        parser.add_argument(
            '--no-pagination',
            action='store_false',
            dest='pagination',
            help='Disable pagination (return all results)')


class _HelpAction(argparse._HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        # print header
        print(HEADER + '\n')

        parser.print_help()
        print('')
        parser.exit()


class _DetailedHelpAction(argparse._HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        # print header
        print(HEADER + '\n')

        parser.print_help()
        print('\n')
        # retrieve subparsers from parser
        subparsers_actions = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        # iterate and print help for each suparser
        for subparsers_action in subparsers_actions:
            # get all subparsers and print help
            for choice, subparsercmd in subparsers_action.choices.items():
                # split help into lines so we can skip the header
                text = subparsercmd.format_help().split('\n')
                # find first line of command (skip usage and header)
                for i, t in enumerate(text):
                    if t.startswith(choice.title()):
                        break
                # print help command prefix
                print(text[i])

                # print help per sub-commands, we actually assume only one
                subact = [
                    action for action in subparsercmd._actions
                    if isinstance(action, argparse._SubParsersAction)
                ]
                # per action print all parameters
                for j, t in enumerate(text[i + 2:]):
                    print(t)
                    k = t.split()
                    if not k:
                        continue
                    try:
                        subc = subact[0].choices[k[0]]
                    except KeyError:
                        continue
                    # hack so we can control formatting in one place
                    # otherwise we need to update all
                    # the parsers when we create them
                    subc.formatter_class = lambda prog: argparse.HelpFormatter(prog, width=120)
                    subchelp = subc.format_help().split('\n')
                    # skip until we reach "optional arguments:"
                    for si, st in enumerate(subchelp):
                        if st.startswith('optional arguments:'):
                            break
                    # print help with single tab indent
                    for st in subchelp[si + 1:]:
                        print('\t %s' % st)
        parser.exit()


def base_arguments(top_parser):
    top_parser.register('action', 'parsers', AliasedSubParsersAction)
    top_parser.add_argument('-h', action=_HelpAction, help='Displays summary of all commands')
    top_parser.add_argument(
        '--help',
        action=_DetailedHelpAction,
        help='Detailed help of command line interface')
    top_parser.add_argument(
        '--version',
        action='version',
        version='CLEARML-AGENT version %s' % Session.version,
        help='CLEARML-AGENT version number')
    top_parser.add_argument(
        '--config-file',
        help='Use a different configuration file (default: "{}")'.format(definitions.CONFIG_FILE))
    top_parser.add_argument('--debug', '-d', action='store_true', help='print debug information')


def bound_number_type(minimum=None, maximum=None):
    """
    bound_number_type

    Creates a bounded integer "type" (validator function)
    for use with argparse.ArgumentParser.add_argument.
    At least one of ``minimum`` and ``maximum`` must be passed.

    :param minimum: maximum allowed value
    :param maximum: minimum allowed value
    """
    if minimum is maximum is None:
        raise ValueError('either "minimum" or "maximum" must be provided')

    def bound_int(arg):
        num = int(arg)
        if minimum is not None and num < minimum:
            raise argparse.ArgumentTypeError('minimum value is {}'.format(minimum))
        if maximum is not None and num > maximum:
            raise argparse.ArgumentTypeError('maximum value is {}'.format(minimum))
        return num

    return bound_int


def real_path_type(string):
    path = Path(string).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError('"{}": No such file or directory'.format(path))
    return path


class ObjectID(object):

    def __init__(self, name, service=None):
        self.name = name
        self.service = service


def foreign_object_id(service):
    return partial(ObjectID, service=service)

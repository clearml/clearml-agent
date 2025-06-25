import argparse
from textwrap import dedent

from clearml_agent.helper.base import warning, is_windows_platform
from clearml_agent.interface.base import foreign_object_id


class DeprecatedFlag(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        warning('argument "{}" is deprecated'.format(option_string))


WORKER_ARGS = {
    '-O': {
        'help': 'Compile optimized pyc code (see python documentation). Repeat for more optimization.',
        'action': 'count',
        'default': 0,
        'dest': 'optimization',
    },
    '--git-user': {
        'help': 'git username for repository access',
    },
    '--git-pass': {
        'help': 'git password (personal access tokens) for repository access',
    },
    '--log-level': {
        'help': 'SDK log level',
        'choices': ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL'],
        'type': lambda x: x.upper(),
        'default': 'INFO',
    },
    '--gpus': {
        'help': 'Specify active GPUs for the daemon to use (docker / virtual environment), '
                'Equivalent to setting NVIDIA_VISIBLE_DEVICES '
                'Examples: --gpus 0 or --gpu 0,1,2 or --gpus all',
        'group': 'Docker support',
    },
    '--cpu-only': {
        'help': 'Disable GPU access for the daemon, only use CPU in either docker or virtual environment',
        'action': 'store_true',
        'group': 'Docker support',
    },
}

DAEMON_ARGS = dict({
    '--polling-interval': {
        'help': 'Polling interval in seconds. Minimum is 5 (default 5)',
        'type': int,
        'default': 5,
    },
    '--foreground': {
        'help': 'Pipe full log to stdout/stderr, should not be used if running in background',
        'action': 'store_true',
    },
    '--docker': {
        'help': 'Run execution task inside a docker (v19.03 and above). Optional args <image> <arguments> or '
                'specify default docker image in agent.default_docker.image / agent.default_docker.arguments '
                'use --gpus/--cpu-only (or set NVIDIA_VISIBLE_DEVICES) to limit gpu visibility for docker',
        'nargs': '*',
        'default': False,
        'group': 'Docker support',
    },
    '--force-current-version': {
        'help': 'Force clearml-agent to use the current clearml-agent version when running in the docker',
        'action': 'store_true',
        'group': 'Docker support',
    },
    '--queue': {
        'help': 'Queue ID(s)/Name(s) to pull tasks from (\'default\' queue).'
                ' Note that the queue list order determines priority, with the first listed queue having the'
                ' highest priority. To change this behavior, use --order-fairness to pull from each queue in a'
                ' round-robin order',
        'nargs': '+',
        'default': tuple(),
        'dest': 'queues',
        'type': foreign_object_id('queues'),
    },
    '--order-fairness': {
        'help': 'Pull from each queue in a round-robin order, instead of priority order.',
        'action': 'store_true',
    },
    '--standalone-mode': {
        'help': 'Do not use any network connects, assume everything is pre-installed',
        'action': 'store_true',
    },
    '--services-mode': {
        'help': 'Launch multiple long-term docker services. Implies docker & cpu-only flags.',
        'nargs': '?',
        'const': -1,
        'type': int,
        'default': None,
    },
    '--child-report-tags': {
        'help': 'List of tags to send with the status reports from the worker that runs a task',
        'nargs': '+',
        'type': str,
        'default': None,
    },
    '--create-queue': {
        'help': 'Create requested queue if it does not exist already.',
        'action': 'store_true',
    },
    '--detached': {
        'help': 'Detached mode, run agent in the background',
        'action': 'store_true',
        'aliases': ['-d'],
    },
    '--stop': {
        'help': 'Stop the running agent (based on the same set of arguments). '
                'Optional: provide a list of specific local worker IDs to stop',
        'nargs': '*',
        'default': False,
    },
    '--dynamic-gpus': {
        'help': 'Allow to dynamically allocate gpus based on queue properties, '
                'configure with \'--queue <queue_name>=<num_gpus>\'.'
                ' Example: \'--dynamic-gpus --gpus 0-3 --queue dual_gpus=2 single_gpu=1\'.'
                ' Example Opportunistic: \'--dynamic-gpus --gpus 0-3 --queue dual_gpus=2 max_quad_gpus=1-4\'.'
                ' Note that the queue list order determines priority, with the first listed queue having the'
                ' highest priority. To change this behavior, use --order-fairness to pull from each queue in a'
                ' round-robin order',
        'action': 'store_true',
    },
    '--uptime': {
        'help': 'Specify uptime for clearml-agent in "<hours> <days>" format. for example, use "17-20 TUE" to set '
                'Tuesday\'s uptime to 17-20'
                'Note: Make sure to have only one of uptime/downtime configuration and not both.',
        'nargs': '*',
        'default': None,
    },
    '--downtime': {
        'help': 'Specify downtime for clearml-agent in "<hours> <days>" format. for example, use "09-13 TUE" to set '
                'Tuesday\'s downtime to 09-13'
                'Note: Make sure to have only on of uptime/downtime configuration and not both.',
        'nargs': '*',
        'default': None,
    },
    '--status': {
        'help': 'Print the worker\'s schedule (uptime properties, server\'s runtime properties and listening queues)',
        'action': 'store_true',
    },
    '--use-owner-token': {
        'help': 'Run tasks under the identity of each task\'s owner: all calls made by the task code during execution '
                'will use the owner\'s credentials instead of the agent\'s. This feature requires the agent to use a '
                'ClearML Enterprise Server.',
        'action': 'store_true',
    }
}, **WORKER_ARGS)

COMMANDS = {
    'execute': {
        'help': 'Build & Execute a selected experiment',
        'args': dict({
            '--id': {
                'help': 'Task ID to run',
                'required': True,
                'dest': 'task_id',
                'type': foreign_object_id('tasks'),
            },
            '--log-file': {
                'help': 'Output task execution (stdout/stderr) into text file',
            },
            '--disable-monitoring': {
                'help': 'Disable logging & monitoring (stdout is still visible)',
                'action': 'store_true',
            },
            '--full-monitoring': {
                'help': 'Full environment setup log & task logging & monitoring (stdout is still visible)',
                'action': 'store_true',
            },
            '--require-queue': {
                'help': 'If the specified task is not queued (in any Queue), the execution will fail. '
                        '(Used for 3rd party scheduler integration, e.g. K8s, SLURM, etc.)',
                'action': 'store_true',
            },
            '--standalone-mode': {
                'help': 'Do not use any network connects, assume everything is pre-installed',
                'action': 'store_true',
            },
            '--docker': {
                'help': 'Run execution task inside a docker (v19.03 and above). Optional args <image> <arguments> or '
                        'specify default docker image in agent.default_docker.image / agent.default_docker.arguments '
                        'use --gpus/--cpu-only (or set NVIDIA_VISIBLE_DEVICES) to limit gpu visibility for docker',
                'nargs': '*',
                'default': False,
            },
            '--clone': {
                'help': 'Clone the experiment before execution, and execute the cloned experiment',
                'action': 'store_true',
            },
        }, **WORKER_ARGS),
    },
    'build': {
        'help': 'Build selected experiment environment '
                '(including pip packages, cloned code and git diff)\n'
                'Used mostly for debugging purposes',
        'args': dict({
            '--id': {
                'help': 'Task ID to build',
                'required': True,
                'dest': 'task_id',
                'type': foreign_object_id('tasks'),
            },
            '--target': {
                'help': 'Where to build the task\'s virtual environment and source code. '
                        'When used with --docker, target docker image name to create',
            },
            '--install-globally': {
                'help': 'Install required python packages before creating the virtual environment used to execute an '
                        'experiment, and use the \'agent.package_manager.system_site_packages\' virtual env option. '
                        'Note: when --docker is used, install-globally is always true',
                'action': 'store_true',
            },
            '--docker': {
                'help': 'Build the experiment inside a docker (v19.03 and above). Optional args <image> <arguments> or '
                'specify default docker image in agent.default_docker.image / agent.default_docker.arguments '
                'use --gpus/--cpu-only (or set NVIDIA_VISIBLE_DEVICES) to limit gpu visibility for docker',
                'nargs': '*',
                'default': False,
            },
            '--force-docker': {
                'help': 'Force using the agent-specified docker image (either explicitly in the --docker argument or '
                        'using the agent\'s default docker image). If provided, the agent will not use any docker '
                        'container information stored on the task itself (default False)',
                'default': False,
                'action': 'store_true',
            },
            '--python-version': {
                'help': 'Virtual environment python version to use',
            },
            '--entry-point': {
                'help': 'Run the task in the new docker. There are two options:\nEither add "reuse_task" to run the '
                'given task in the docker, or "clone_task" to first clone the given task and then run it in the docker',
                'default': False,
                'choices': ['reuse_task', 'clone_task'],
            }
        }, **WORKER_ARGS),
    },
    'list': {
        'help': 'List all worker machines and status',
    },
    'daemon': {
        'help': 'Start ClearML-Agent daemon worker',
        'args': DAEMON_ARGS,
    },
    'config': {
        'help': 'Check daemon configuration and print it',
    },
    'init': {
        'help': 'ClearML-Agent configuration wizard',
    }
}

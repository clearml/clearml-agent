import abc
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from hashlib import md5
from os import environ
from random import random
from threading import Lock
from typing import Text, Sequence, Mapping, Iterable, TypeVar, Callable, Tuple, Optional

from .._vendor import attr
from .._vendor.furl import furl
from .._vendor.pathlib2 import Path

from .._vendor import six

from clearml_agent.definitions import ENV_AGENT_GIT_USER, ENV_AGENT_GIT_PASS, ENV_AGENT_GIT_HOST, ENV_GIT_CLONE_VERBOSE
from clearml_agent.helper.console import ensure_text, ensure_binary
from clearml_agent.errors import CommandFailedError
from clearml_agent.helper.base import (
    select_for_platform,
    rm_tree,
    ExecutionInfo,
    normalize_path,
    create_file_if_not_exists, safe_remove_file,
)
from clearml_agent.helper.os.locks import FileLock
from clearml_agent.helper.process import DEVNULL, Argv, PathLike, COMMAND_SUCCESS, find_executable
from clearml_agent.session import Session


class VcsFactory(object):
    """
    Creates VCS instances
    """

    GIT_SUFFIX = ".git"

    @classmethod
    def create(cls, session, execution_info, location):
        # type: (Session, ExecutionInfo, PathLike) -> VCS
        """
        Create a VCS instance for config and url
        :param session: program session
        :param execution_info: task ExecutionInfo
        :param location: (desired) clone location
        """
        url = execution_info.repository
        # We only support git, hg is deprecated
        is_git = True
        # is_git = url.endswith(cls.GIT_SUFFIX)
        vcs_cls = Git if is_git else Hg
        revision = (
            execution_info.version_num
            or execution_info.tag
            or vcs_cls.remote_branch_name(execution_info.branch or vcs_cls.main_branch)
        )
        return vcs_cls(session, url, location, revision)


# noinspection PyUnresolvedReferences
@attr.s
class RepoInfo(object):
    """
    Cloned repository information
    :param type: VCS type
    :param url: repository url
    :param branch: revision branch
    :param commit: revision number
    :param root: clone location path
    """

    type = attr.ib(type=str)
    url = attr.ib(type=str)
    branch = attr.ib(type=str)
    commit = attr.ib(type=str)
    root = attr.ib(type=str)


RType = TypeVar("RType")


@six.add_metaclass(abc.ABCMeta)
class VCS(object):

    """
    Provides overloaded utilities for handling repositories of different types
    """

    # additional environment variables for VCS commands
    COMMAND_ENV = {}

    PATCH_ADDED_FILE_RE = re.compile(r"^--- a/(?P<path>.*)")

    def __init__(self, session, url, location, revision):
        # type: (Session, Text, PathLike, Text) -> ()
        """
        Create a VCS instance for config and url
        :param session: program session
        :param url: repository url
        :param location: (desired) clone location
        :param revision: desired clone revision
        """
        self.session = session
        self.log = self.session.get_logger(
            "{}.{}".format(__name__, type(self).__name__)
        )
        self.url = url
        self.location = Text(location)
        self._revision = revision
        self.log = self.session.get_logger(__name__)

    @property
    def url_with_auth(self):
        """
        Return URL with configured user/password
        """
        return self.add_auth(self.session.config, self.url)

    @property
    def url_without_auth(self):
        """
        Return URL without configured user/password
        """
        return self.add_auth(self.session.config, self.url, reset_auth=True)

    @abc.abstractmethod
    def executable_name(self):
        """
        Name of command executable
        """
        pass

    @abc.abstractmethod
    def main_branch(self):
        """
        Name of default/main branch
        """
        pass

    @abc.abstractmethod
    def checkout_flags(self):
        # type: () -> Sequence[Text]
        """
        Command-line flags for checkout command
        """
        pass

    @abc.abstractmethod
    def patch_base(self):
        # type: () -> Sequence[Text]
        """
        Command and flags for applying patches
        """
        pass

    def patch(self, location, patch_content):
        # type: (PathLike, Text) -> bool
        """
        Apply patch repository at `location`
        """
        self.log.info("applying diff to %s" % location)

        # noinspection PyBroadException
        try:
            for match in filter(
                None, map(self.PATCH_ADDED_FILE_RE.match, patch_content.splitlines())
            ):
                file_path = None
                # noinspection PyBroadException
                try:
                    file_path = normalize_path(location, match.group("path"))
                    create_file_if_not_exists(file_path)
                except Exception:
                    if file_path:
                        self.log.warning("failed creating file for git diff (%s)" % file_path)
        except Exception:
            pass

        return_code, errors = self.call_with_stdin(
            patch_content, *self.patch_base, cwd=location
        )
        if return_code:
            self.log.error("Failed applying diff")
            lines = errors.splitlines()
            if any(l for l in lines if "no such file or directory" in l.lower()):
                self.log.warning(
                    "NOTE: files were not found when applying diff, perhaps you forgot to push your changes?"
                )
            return False
        else:
            self.log.info("successfully applied uncommitted changes")
        return True

    def clone_flags(self):
        """Command-line flags for clone command"""
        return tuple()

    @abc.abstractmethod
    def executable_not_found_error_help(self):
        # type: () -> Text
        """
        Instructions for when executable is not found
        """
        pass

    @staticmethod
    def remote_branch_name(branch):
        # type: (Text) -> Text
        """
        Creates name of remote branch from name of local/ambiguous branch.
        Returns same name by default.
        """
        return branch

    # parse scp-like git ssh URLs, e.g: git@host:user/project.git
    SSH_URL_GIT_SYNTAX = re.compile(
        r"""
        ^
        (?:(?P<user>{regular}*?)@)?
        (?P<host>{regular}*?)
        :
        (?P<path>{regular}.*)?
        $
        """.format(
            regular=r"[^/@:#]"
        ),
        re.VERBOSE,
    )

    @classmethod
    def replace_ssh_url(cls, url):
        # type: (Text) -> Text
        """
        Replace SSH URL with HTTPS URL when applicable
        """

        def get_username(user_, password=None):
            """
            Remove special SSH users hg/git
            """
            return (
                None
                if user_ and user_.lower() in ["hg", "git"] and not password
                else user_
            )

        match = cls.SSH_URL_GIT_SYNTAX.match(url)
        if match:
            user, host, path = match.groups()
            return (
                furl()
                .set(scheme="https", username=get_username(user), host=host, path=path)
                .url
            )
        parsed_url = furl(url)
        if parsed_url.scheme == "ssh":
            return parsed_url.set(
                scheme="https",
                username=get_username(
                    parsed_url.username, password=parsed_url.password
                ),
            ).url
        return url

    @classmethod
    def replace_http_url(cls, url, port=None, username=None):
        # type: (Text, Optional[int], Optional[str]) -> Text
        """
        Replace HTTPS URL with SSH URL when applicable
        """
        parsed_url = furl(url)
        if parsed_url.scheme == "https":
            parsed_url.scheme = "ssh"
            parsed_url.username = username or "git"
            parsed_url.password = None
            # make sure there is no port in the final url (safe_furl support)
            # the original port was an https port, and we do not know if there is a different ssh port,
            # so we have to clear the original port specified (https) and use the default ssh schema port.
            parsed_url.port = port or None
            url = parsed_url.url
        return url

    @classmethod
    def rewrite_ssh_url(cls, url, port=None, username=None):
        # type: (Text, Optional[int], Optional[str]) -> Text
        """
        Rewrite SSH URL with custom port and username
        """
        parsed_url = furl(url)
        if parsed_url.scheme == "ssh":
            parsed_url.username = username or "git"
            parsed_url.port = port or None
            return parsed_url.url

    def _set_ssh_url(self):
        """
        Replace instance URL with SSH substitution result and report to log.
        According to ``man ssh-add``, ``SSH_AUTH_SOCK`` must be set in order for ``ssh-add`` to work.
        """
        if self.session.config.get('agent.force_git_ssh_protocol', None) and self.url:
            parsed_url = furl(self.url)
            # only apply to a specific domain (if requested)
            config_domain = \
                ENV_AGENT_GIT_HOST.get() or self.session.config.get("agent.git_host", None)
            if config_domain and config_domain != parsed_url.host:
                return
            if parsed_url.scheme == "https":
                new_url = self.replace_http_url(
                    self.url,
                    port=self.session.config.get('agent.force_git_ssh_port', None),
                    username=self.session.config.get('agent.force_git_ssh_user', None)
                )
                if new_url != self.url:
                    print("Using SSH credentials - replacing https url '{}' with ssh url '{}'".format(
                        self.url, new_url))
                    self.url = new_url
                return

            # rewrite ssh URLs only if either ssh port or ssh user are forced in config
            # TODO: fix, when url is in the form of `git@domain.com:user/project.git` we will fail to get scheme
            # need to add ssh:// and replace first ":" with / , unless port is specified
            if parsed_url.scheme == "ssh" and (
                self.session.config.get('agent.force_git_ssh_port', None) or
                self.session.config.get('agent.force_git_ssh_user', None)
            ):
                new_url = self.rewrite_ssh_url(
                    self.url,
                    port=self.session.config.get('agent.force_git_ssh_port', None),
                    username=self.session.config.get('agent.force_git_ssh_user', None)
                )
                if new_url != self.url:
                    print("Using SSH credentials - ssh url '{}' with ssh url '{}'".format(
                        self.url, new_url))
                    self.url = new_url
                return
            elif parsed_url.scheme == "ssh":
                return

        if not self.session.config.agent.translate_ssh:
            return

        # if we have git_user / git_pass replace ssh credentials with https authentication
        if (ENV_AGENT_GIT_USER.get() or self.session.config.get('agent.git_user', None)) and \
                (ENV_AGENT_GIT_PASS.get() or self.session.config.get('agent.git_pass', None)):

            # only apply to a specific domain (if requested)
            config_domain = \
                ENV_AGENT_GIT_HOST.get() or self.session.config.get("agent.git_host", None)
            if config_domain:
                if config_domain != furl(self.url).host:
                    # bail out here if we have a git_host configured and it's different than the URL host
                    # however, we should make sure this is not an ssh@ URL that furl failed to parse
                    ssh_git_url_match = self.SSH_URL_GIT_SYNTAX.match(self.url)
                    if not ssh_git_url_match or config_domain != ssh_git_url_match.groupdict().get("host"):
                        # do not replace to ssh url
                        return

            new_url = self.replace_ssh_url(self.url)
            if new_url != self.url:
                print("Using user/pass credentials - replacing ssh url '{}' with https url '{}'".format(
                    self.url, new_url))
                self.url = new_url

    def clone(self, branch=None):
        # type: (Text) -> None
        """
        Clone repository to destination and checking out `branch`.
        If not in debug mode, filter VCS password from output.
        """
        self._set_ssh_url()
        # if we are on linux no need for the full auth url because we use GIT_ASKPASS
        url = self.url_without_auth if self._use_ask_pass else self.url_with_auth
        clone_command = ("clone", url, self.location) + self.clone_flags()
        # clone all branches regardless of when we want to later checkout
        # if branch:
        #    clone_command += ("-b", branch)
        if self.session.debug_mode:
            self.call(*clone_command)
            return

        try:
            self._print_output(self._normalize_output(self.get_stderr(*clone_command)))
        except subprocess.CalledProcessError as e:
            # In Python 3, subprocess.CalledProcessError has a `stderr` attribute,
            # but since stderr is redirect to `subprocess.PIPE` it will appear in the usual `output` attribute
            if e.output:
                e.output = self._normalize_output(e.output)
                self._print_output(e.output)
            raise

    def _normalize_output(self, result):
        """
        Returns result string without user's password.
        NOTE: ``self.get_stderr``'s result might or might not have the same type as ``e.output`` in case of error.
        """
        string_type = (
            ensure_text
            if isinstance(result, six.text_type)
            else ensure_binary
        )
        return result.replace(
            string_type(self.url),
            string_type(furl(self.url).remove(password=True).tostr()),
        )

    @staticmethod
    def _print_output(output):
        print(ensure_text(output))

    def checkout(self):
        # type: () -> None
        """
        Checkout repository at specified revision
        """
        self.call("checkout", self._revision, *self.checkout_flags, cwd=self.location)

    @abc.abstractmethod
    def pull(self):
        # type: () -> None
        """
        Pull remote changes for revision
        """
        pass

    def call(self, *argv, **kwargs):
        """
        Execute argv without stdout/stdin.
        Remove stdin so git/hg can't ask for passwords.
        ``kwargs`` can override all arguments passed to subprocess.
        """
        return self._call_subprocess(subprocess.check_call, argv, **kwargs)

    def call_with_stdin(self, input_, *argv, **kwargs):
        # type: (...) -> Tuple[int, str]
        """
        Run command with `input_` as stdin
        """
        input_ = input_.encode("utf-8")
        # always add extra empty line 
        # (there is no downside, and it solves empty lines issue at end of patch cause corrupt message)
        input_ += b"\n"
        process = self._call_subprocess(
            subprocess.Popen,
            argv,
            **dict(
                kwargs,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        _, stderr = process.communicate(input_)
        if stderr:
            self.log.warning("%s: %s", self._get_vcs_command(argv), stderr)
        return process.returncode, Text(stderr)

    def get_stderr(self, *argv, **kwargs):
        """
        Execute argv without stdout/stdin in <cwd> and get stderr output.
        Remove stdin so git/hg can't ask for passwords.
        ``kwargs`` can override all arguments passed to subprocess.
        """
        process = self._call_subprocess(
            subprocess.Popen, argv, **dict(kwargs, stderr=subprocess.PIPE, stdout=None)
        )
        _, stderr = process.communicate()
        code = process.poll()
        if code == COMMAND_SUCCESS:
            return stderr
        with Argv.normalize_exception(censor_password=True):
            raise subprocess.CalledProcessError(
                returncode=code, cmd=argv, output=stderr
            )

    def _call_subprocess(self, func, argv, **kwargs):
        # type: (Callable[..., RType], Iterable[Text], dict) ->  RType
        cwd = kwargs.pop("cwd", None)
        cwd = cwd and str(cwd)
        kwargs = dict(
            dict(
                censor_password=True,
                cwd=cwd,
                stdin=DEVNULL,
                stdout=DEVNULL,
                env=dict(self.COMMAND_ENV, **environ),
            ),
            **kwargs
        )
        command = self._get_vcs_command(argv)
        self.log.debug("Running: %s", list(command))
        return command.call_subprocess(func, **kwargs)

    def _get_vcs_command(self, argv):
        # type: (Iterable[PathLike]) -> Argv
        return Argv(self.executable_name, *argv)

    @classmethod
    def add_auth(cls, config, url, reset_auth=False):
        """
        Add username and password to URL if missing from URL and present in config.
        Does not modify ssh URLs.

        :param reset_auth: If true remove the user/pass from the URL (default False)
        """
        try:
            parsed_url = furl(url)
        except ValueError:
            return url
        if parsed_url.scheme in ["", "ssh"] or (parsed_url.scheme or '').startswith("git"):
            return parsed_url.url
        config_user = ENV_AGENT_GIT_USER.get() or config.get("agent.{}_user".format(cls.executable_name), None)
        config_pass = ENV_AGENT_GIT_PASS.get() or config.get("agent.{}_pass".format(cls.executable_name), None)
        config_domain = ENV_AGENT_GIT_HOST.get() or config.get("agent.{}_host".format(cls.executable_name), None)
        if (
            (not (parsed_url.username and parsed_url.password))
            and config_user
            and config_pass
            and (not config_domain or config_domain.lower() == parsed_url.host)
        ):
            if reset_auth:
                parsed_url.set(username=None, password=None)
            else:
                parsed_url.set(username=config_user, password=config_pass)
        return parsed_url.url

    @abc.abstractmethod
    def info_commands(self):
        # type: () -> Mapping[Text, Argv]
        """
    `   Mapping from `RepoInfo` attribute name (except `type`) to command which acquires it
        """
        pass

    def get_repository_copy_info(self, path):
        """
        Get `RepoInfo` instance from copy of clone in `path`
        """
        path = Text(path)
        commands_result = {
            name: command.get_output(cwd=path)
            # name: subprocess.check_output(command.split(), cwd=path).decode().strip()
            for name, command in self.info_commands.items()
        }
        return RepoInfo(type=self.executable_name, **commands_result)


class Git(VCS):
    executable_name = "git"
    main_branch = ("master", "main")
    checkout_flags = ("--force",)
    COMMAND_ENV = {
        # do not prompt for password
        "GIT_TERMINAL_PROMPT": "0",
        # do not prompt for ssh key passphrase
        "GIT_SSH_COMMAND": "ssh -oBatchMode=yes",
    }

    def __init__(self, *args, **kwargs):
        super(Git, self).__init__(*args, **kwargs)

        self._use_ask_pass = False if not self.session.config.get('agent.enable_git_ask_pass', True) \
            else sys.platform == "linux"

        try:
            self.call("config", "--global", "--replace-all", "safe.directory", "*", cwd=self.location)
        except:  # noqa
            pass

    @staticmethod
    def remote_branch_name(branch):
        return [
            "origin/{}".format(b) for b in ([branch] if isinstance(branch, str) else branch)
        ]

    def clone_flags(self):
        return (
            "--recursive",
            "--verbose" if ENV_GIT_CLONE_VERBOSE.get() else "--quiet"
        )

    def executable_not_found_error_help(self):
        return 'Cannot find "{}" executable. {}'.format(
            self.executable_name,
            select_for_platform(
                linux="You can install it by running: sudo apt-get install {}".format(
                    self.executable_name
                ),
                windows="You can download it here: {}".format(
                    "https://gitforwindows.org/"
                ),
            ),
        )

    def pull(self):
        self._set_ssh_url()
        self.call("fetch", "--all", "--tags", "--recurse-submodules", cwd=self.location)

    def _git_pass_auth_wrapper(self, func, *args, **kwargs):
        try:
            url_with_auth = furl(self.url_with_auth)
            password = url_with_auth.password if url_with_auth else None
            username = url_with_auth.username if url_with_auth else None
        except:  # noqa
            password = None
            username = None

        # if this is not linux or we do not have a password, just run as is
        if not self._use_ask_pass or not password or not username:
            return func(*args, **kwargs)

        # create the password file
        fp, pass_file = tempfile.mkstemp(prefix='clearml_git_', suffix='.sh')
        os.close(fp)
        with open(pass_file, 'wt') as f:
            # get first letter only (username / password are the argument options)
            # then echo the correct information
            f.writelines([
                '#!/bin/bash\n',
                'c="$1"\n',
                'c="${c%"${c#?}"}"\n',
                'if [ "$c" == "u" ] || [ "$c" == "U" ]; then echo "{}"; else echo "{}"; fi\n'.format(
                    username.replace('"', '\\"'), password.replace('"', '\\"')
                )
            ])
        # mark executable
        st = os.stat(pass_file)
        os.chmod(pass_file, st.st_mode | stat.S_IEXEC)
        # let GIT use it
        self.COMMAND_ENV["GIT_ASKPASS"] = pass_file
        # call git command
        try:
            ret = func(*args, **kwargs)
        finally:
            # delete temp password file
            self.COMMAND_ENV.pop("GIT_ASKPASS", None)
            safe_remove_file(pass_file)

        return ret

    def get_stderr(self, *argv, **kwargs):
        """
        Wrapper with git password authentication
        """
        return self._git_pass_auth_wrapper(super(Git, self).get_stderr, *argv, **kwargs)

    def call_with_stdin(self, *argv, **kwargs):
        """
        Wrapper with git password authentication
        """
        return self._git_pass_auth_wrapper(super(Git, self).call_with_stdin, *argv, **kwargs)

    def call(self, *argv, **kwargs):
        """
        Wrapper with git password authentication
        """
        return self._git_pass_auth_wrapper(super(Git, self).call, *argv, **kwargs)

    def checkout(self):  # type: () -> None
        """
        Checkout repository at specified revision
        """
        revisions = [self._revision] if isinstance(self._revision, str) else self._revision
        for i, revision in enumerate(revisions):
            try:
                self.call("checkout", revision, *self.checkout_flags, cwd=self.location)
                break
            except subprocess.CalledProcessError:
                if i == len(revisions) - 1:
                    raise

        try:
            self.call("submodule", "update", "--recursive", cwd=self.location)
        except:  # noqa
            pass

    info_commands = dict(
        url=Argv(executable_name, "ls-remote", "--get-url", "origin"),
        branch=Argv(executable_name, "rev-parse", "--abbrev-ref", "HEAD"),
        commit=Argv(executable_name, "rev-parse", "HEAD"),
        root=Argv(executable_name, "rev-parse", "--show-toplevel"),
    )

    patch_base = ("apply", "--unidiff-zero", )


class Hg(VCS):
    executable_name = "hg"
    main_branch = "default"
    checkout_flags = ("--clean",)
    patch_base = ("import", "--no-commit")

    def executable_not_found_error_help(self):
        return 'Cannot find "{}" executable. {}'.format(
            self.executable_name,
            select_for_platform(
                linux="You can install it by running: sudo apt-get install {}".format(
                    self.executable_name
                ),
                windows="You can download it here: {}".format(
                    "https://www.mercurial-scm.org/wiki/Download"
                ),
            ),
        )

    def pull(self):
        self.call(
            "pull",
            self.url_with_auth,
            cwd=self.location,
            *(("-r", self._revision) if self._revision else ())
        )

    info_commands = dict(
        url=Argv(executable_name, "paths", "--verbose"),
        branch=Argv(executable_name, "--debug", "id", "-b"),
        commit=Argv(executable_name, "--debug", "id", "-i"),
        root=Argv(executable_name, "root"),
    )


def clone_repository_cached(session, execution, destination):
    # type: (Session, ExecutionInfo, Path) -> Tuple[VCS, RepoInfo]
    """
    Clone a remote repository.
    :param execution: execution info
    :param destination: directory to clone to (in which a directory for the repository will be created)
    :param session: program session
    :return: repository information
    :raises: CommandFailedError if git/hg is not installed
    """
    # mock lock
    repo_lock = Lock()
    repo_lock_timeout_sec = 300
    repo_url = execution.repository or ''  # type: str
    parsed_url = furl(repo_url)
    no_password_url = parsed_url.copy().remove(password=True).url

    clone_folder_name = Path(str(furl(repo_url).path)).name  # type: str
    clone_folder = Path(destination) / clone_folder_name

    standalone_mode = session.config.get("agent.standalone_mode", False)
    if standalone_mode:
        cached_repo_path = clone_folder
    else:
        vcs_cache_path = Path(session.config["agent.vcs_cache.path"]).expanduser()
        repo_hash = md5(ensure_binary(repo_url)).hexdigest()
        # create lock
        repo_lock = FileLock(filename=(vcs_cache_path / '{}.lock'.format(repo_hash)).as_posix())
        # noinspection PyBroadException
        try:
            repo_lock.acquire(timeout=repo_lock_timeout_sec)
        except BaseException:
            print('Could not lock cache folder "{}" (timeout {} sec), using temp vcs cache.'.format(
                clone_folder_name, repo_lock_timeout_sec))
            repo_hash = '{}_{}'.format(repo_hash, str(random()).replace('.', ''))
            # use mock lock for the context
            repo_lock = Lock()
        # select vcs cache folder
        cached_repo_path = vcs_cache_path / "{}.{}".format(clone_folder_name, repo_hash) / clone_folder_name

    with repo_lock:
        vcs = VcsFactory.create(
            session, execution_info=execution, location=cached_repo_path
        )
        if not find_executable(vcs.executable_name):
            raise CommandFailedError(vcs.executable_not_found_error_help())

        if not standalone_mode:
            if session.config["agent.vcs_cache.enabled"] and cached_repo_path.exists():
                print('Using cached repository in "{}"'.format(cached_repo_path))

            else:
                print("cloning: {}".format(no_password_url))
                rm_tree(cached_repo_path)
                # We clone the entire repository, not a specific branch
                vcs.clone()  # branch=execution.branch)

            print("pulling git")
            try:
                vcs.pull()
            except Exception as ex:
                print("git pull failed: {}".format(ex))
                if (
                        session.config.get("agent.vcs_cache.enabled", False) and
                        session.config.get("agent.vcs_cache.clone_on_pull_fail", False)
                ):
                    print("pulling git failed, re-cloning: {}".format(no_password_url))
                    rm_tree(cached_repo_path)
                    vcs.clone()
                else:
                    raise ex
            print("pulling git completed")

            rm_tree(destination)
            shutil.copytree(Text(cached_repo_path), Text(clone_folder),
                            symlinks=select_for_platform(linux=True, windows=False),
                            ignore_dangling_symlinks=True)
            if not clone_folder.is_dir():
                raise CommandFailedError(
                    "copying of repository failed: from {} to {}".format(
                        cached_repo_path, clone_folder
                    )
                )

            # checkout in the newly copy destination
            vcs.location = Text(clone_folder)
            vcs.checkout()

    repo_info = vcs.get_repository_copy_info(clone_folder)

    # make sure we have no user/pass in the returned repository structure
    repo_info = attr.evolve(repo_info, url=no_password_url)

    return vcs, repo_info


def fix_package_import_diff_patch(entry_script_file):
    # noinspection PyBroadException
    try:
        with open(entry_script_file, 'rt') as f:
            lines = f.readlines()
    except Exception:
        return
    # make sure we are the first import (i.e. we patched the source code)
    if len(lines or []) < 2 or not lines[0].strip().startswith('from clearml ') or 'Task.init' not in lines[1]:
        return

    original_lines = lines
    # skip over the first two lines, they are ours
    # then skip over empty or comment lines
    lines = [(i, line.split('#', 1)[0].rstrip()) for i, line in enumerate(lines)
             if i >= 2 and line.strip('\r\n\t ') and not line.strip().startswith('#')]

    # remove triple quotes ' """ '
    nested_c = -1
    skip_lines = []
    for i, line_pair in enumerate(lines):
        for _ in line_pair[1].split('"""')[1:]:
            if nested_c >= 0:
                skip_lines.extend(list(range(nested_c, i+1)))
                nested_c = -1
            else:
                nested_c = i
    # now select all the
    lines = [pair for i, pair in enumerate(lines) if i not in skip_lines]

    from_future = re.compile(r"^from[\s]*__future__[\s]*")
    import_future = re.compile(r"^import[\s]*__future__[\s]*")
    # test if we have __future__ import
    found_index = -1
    for a_i, (_, a_line) in enumerate(lines):
        if found_index >= a_i:
            continue
        if from_future.match(a_line) or import_future.match(a_line):
            found_index = a_i
            # check the last import block
            i, line = lines[found_index]
            # wither we have \\ character at the end of the line or the line is indented
            parenthesized_lines = '(' in line and ')' not in line
            while line.endswith('\\') or parenthesized_lines:
                found_index += 1
                i, line = lines[found_index]
                if ')' in line:
                    break

        else:
            break

    # no imports found
    if found_index < 0:
        return

    # now we need to move back the patched two lines
    entry_line = lines[found_index][0]
    new_lines = original_lines[2:entry_line + 1] + original_lines[0:2] + original_lines[entry_line + 1:]
    # noinspection PyBroadException
    try:
        with open(entry_script_file, 'wt') as f:
            f.writelines(new_lines)
    except Exception:
        return


def _locate_future_import(lines):
    # type: (list[str]) -> int
    """
    :param lines: string lines of a python file
    :return: line index of the last __future_ import. return -1 if no __future__ was found
    """
    # skip over the first two lines, they are ours
    # then skip over empty or comment lines
    lines = [(i, line.split('#', 1)[0].rstrip()) for i, line in enumerate(lines)
             if line.strip('\r\n\t ') and not line.strip().startswith('#')]

    # remove triple quotes ' """ '
    nested_c = -1
    skip_lines = []
    for i, line_pair in enumerate(lines):
        for _ in line_pair[1].split('"""')[1:]:
            if nested_c >= 0:
                skip_lines.extend(list(range(nested_c, i + 1)))
                nested_c = -1
            else:
                nested_c = i
    # now select all the
    lines = [pair for i, pair in enumerate(lines) if i not in skip_lines]

    from_future = re.compile(r"^from[\s]*__future__[\s]*")
    import_future = re.compile(r"^import[\s]*__future__[\s]*")
    # test if we have __future__ import
    found_index = -1
    for a_i, (_, a_line) in enumerate(lines):
        if found_index >= a_i:
            continue
        if from_future.match(a_line) or import_future.match(a_line):
            found_index = a_i
            # check the last import block
            i, line = lines[found_index]
            # wither we have \\ character at the end of the line or the line is indented
            parenthesized_lines = '(' in line and ')' not in line
            while line.endswith('\\') or parenthesized_lines:
                found_index += 1
                i, line = lines[found_index]
                if ')' in line:
                    break

        else:
            break

    return found_index if found_index < 0 else lines[found_index][0]


def patch_add_task_init_call(local_filename):
    if not local_filename or not Path(local_filename).is_file() or not str(local_filename).lower().endswith(".py"):
        return

    idx_a = 0
    # find the right entry for the patch if we have a local file (basically after __future__
    try:
        with open(local_filename, 'rt') as f:
            lines = f.readlines()
    except Exception as ex:
        print("Failed patching entry point file {}: {}".format(local_filename, ex))
        return

    future_found = _locate_future_import(lines)
    if future_found >= 0:
        idx_a = future_found + 1

    # check if we have not already patched it, no need to add another one
    if len(lines or []) >= idx_a+2 and lines[idx_a].strip().startswith('from clearml ') and 'Task.init' in lines[idx_a+1]:
        print("File {} already patched with Task.init()".format(local_filename))
        return

    patch = [
        "from clearml import Task\n",
        "(__name__ != \"__main__\") or Task.init()\n",
    ]
    lines = lines[:idx_a] + patch + lines[idx_a:]
    # noinspection PyBroadException
    try:
        with open(local_filename, 'wt') as f:
            f.writelines(lines)
    except Exception as ex:
        print("Failed patching entry point file {}: {}".format(local_filename, ex))
        return

    print("Force clearml Task.init patch adding to entry point script: {}".format(local_filename))

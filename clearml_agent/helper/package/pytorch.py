from __future__ import unicode_literals

import re
import sys
import platform
from ..._vendor.furl import furl
import urllib.parse
from operator import itemgetter
from html.parser import HTMLParser
from typing import Text, Optional, Dict

from ..._vendor import attr
import requests

from ..._vendor import six
from .requirements import (
    SimpleSubstitution, FatalSpecsResolutionError, SimpleVersion, MarkerRequirement,
    compare_version_rules, )
from ...definitions import ENV_PACKAGE_PYTORCH_RESOLVE
from ...external.requirements_parser.requirement import Requirement

OS_TO_WHEEL_NAME = {"linux": "linux_x86_64", "windows": "win_amd64"}


def os_to_wheel_name(x):
    return OS_TO_WHEEL_NAME[x]


def fix_version(version):
    def replace(nums, prerelease):
        if prerelease:
            return "{}-{}".format(nums, prerelease)
        return nums

    return re.sub(
        r"(\d+(?:\.\d+){,2})(?:\.(.*))?",
        lambda match: replace(*match.groups()),
        version,
    )


class LinksHTMLParser(HTMLParser):
    def __init__(self):
        super(LinksHTMLParser, self).__init__()
        self.links = []

    def handle_data(self, data):
        if data and data.strip():
            self.links += [data]


@attr.s
class PytorchWheel(object):
    os_name = attr.ib(type=str, converter=os_to_wheel_name)
    cuda_version = attr.ib(converter=lambda x: "cu{}".format(x) if x else "cpu")
    python = attr.ib(type=str, converter=lambda x: str(x).replace(".", ""))
    torch_version = attr.ib(type=str, converter=fix_version)

    url_template_prefix = "http://download.pytorch.org/whl/"
    url_template = "{0.cuda_version}/torch-{0.torch_version}" \
                   "-cp{0.python}-cp{0.python}m{0.unicode}-{0.os_name}.whl"

    def __attrs_post_init__(self):
        self.unicode = "u" if self.python.startswith("2") else ""

    def make_url(self):
        # type: () -> Text
        return (self.url_template_prefix + self.url_template).format(self)


class PytorchResolutionError(FatalSpecsResolutionError):
    pass


class SimplePytorchRequirement(SimpleSubstitution):
    name = "torch"

    packages = ("torch", "torchvision", "torchaudio")

    page_lookup_template = 'https://download.pytorch.org/whl/cu{}/torch_stable.html'
    nightly_page_lookup_template = 'https://download.pytorch.org/whl/nightly/cu{}/torch_nightly.html'
    torch_page_lookup = {
        0: 'https://download.pytorch.org/whl/cpu/torch_stable.html',
        80: 'https://download.pytorch.org/whl/cu80/torch_stable.html',
        90: 'https://download.pytorch.org/whl/cu90/torch_stable.html',
        92: 'https://download.pytorch.org/whl/cu92/torch_stable.html',
        100: 'https://download.pytorch.org/whl/cu100/torch_stable.html',
        101: 'https://download.pytorch.org/whl/cu101/torch_stable.html',
        102: 'https://download.pytorch.org/whl/cu102/torch_stable.html',
        110: 'https://download.pytorch.org/whl/cu110/torch_stable.html',
    }

    def __init__(self, *args, **kwargs):
        super(SimplePytorchRequirement, self).__init__(*args, **kwargs)
        self._matched = False

    def match(self, req):
        # match both any of out packages
        return req.name in self.packages

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Get rid of +cpu +cu?? etc.
        try:
            req.specs[0] = (req.specs[0][0], req.specs[0][1].split('+')[0])
        except:
            pass
        self._matched = True
        return Text(req)

    def matching_done(self, reqs, package_manager):
        # type: (Sequence[MarkerRequirement], object) -> ()
        if not self._matched:
            return
        # TODO: add conda channel support
        from .pip_api.system import SystemPip
        if package_manager and isinstance(package_manager, SystemPip):
            extra_url, _ = self.get_torch_page(self.cuda_version)
            package_manager.add_extra_install_flags(('-f', extra_url))

    @classmethod
    def get_torch_page(cls, cuda_version, nightly=False):
        # noinspection PyBroadException
        try:
            cuda = int(cuda_version)
        except Exception:
            cuda = 0

        if nightly:
            for c in range(cuda, max(-1, cuda-15), -1):
                # then try the nightly builds, it might be there...
                torch_url = cls.nightly_page_lookup_template.format(c)
                # noinspection PyBroadException
                try:
                    if requests.get(torch_url, timeout=10).ok:
                        print('Torch nightly CUDA {} download page found'.format(c))
                        cls.torch_page_lookup[c] = torch_url
                        return cls.torch_page_lookup[c], c
                except Exception:
                    pass
            return

        # first check if key is valid
        if cuda in cls.torch_page_lookup:
            return cls.torch_page_lookup[cuda], cuda

        # then try a new cuda version page
        for c in range(cuda, max(-1, cuda-15), -1):
            torch_url = cls.page_lookup_template.format(c)
            # noinspection PyBroadException
            try:
                if requests.get(torch_url, timeout=10).ok:
                    print('Torch CUDA {} download page found'.format(c))
                    cls.torch_page_lookup[c] = torch_url
                    return cls.torch_page_lookup[c], c
            except Exception:
                pass

        keys = sorted(cls.torch_page_lookup.keys(), reverse=True)
        for k in keys:
            if k <= cuda:
                return cls.torch_page_lookup[k], k
        # return default - zero
        return cls.torch_page_lookup[0], 0


class PytorchRequirement(SimpleSubstitution):

    name = "torch"
    packages = ("torch", "torchvision", "torchaudio", "torchcsprng", "torchtext")

    extra_index_url_template = 'https://download.pytorch.org/whl/cu{}/'
    nightly_extra_index_url_template = 'https://download.pytorch.org/whl/nightly/cu{}/'
    torch_index_url_lookup = {}
    resolver_types = ("pip", "direct", "none")

    def __init__(self, *args, **kwargs):
        os_name = kwargs.pop("os_override", None)
        super(PytorchRequirement, self).__init__(*args, **kwargs)
        self.log = self._session.get_logger(__name__)
        self.package_manager = self.config["agent.package_manager.type"].lower()
        self.os = os_name or self.get_platform()
        self.cuda = None
        self.python_version_string = None
        self.python_major_minor_str = None
        self.python = None
        self._fix_setuptools = None
        self.exceptions = []
        self._original_req = []
        # allow override pytorch lookup pages
        if self.config.get("agent.package_manager.extra_index_url_template", None):
            self.extra_index_url_template = \
                self.config.get("agent.package_manager.extra_index_url_template", None)
        if self.config.get("agent.package_manager.nightly_extra_index_url_template", None):
            self.nightly_extra_index_url_template = \
                self.config.get("agent.package_manager.nightly_extra_index_url_template", None)
        # allow override pytorch lookup pages
        if self.config.get("agent.package_manager.torch_page", None):
            SimplePytorchRequirement.page_lookup_template = \
                self.config.get("agent.package_manager.torch_page", None)
        if self.config.get("agent.package_manager.torch_nightly_page", None):
            SimplePytorchRequirement.nightly_page_lookup_template = \
                self.config.get("agent.package_manager.torch_nightly_page", None)
        if self.config.get("agent.package_manager.torch_url_template_prefix", None):
            PytorchWheel.url_template_prefix = \
                self.config.get("agent.package_manager.torch_url_template_prefix", None)
        if self.config.get("agent.package_manager.torch_url_template", None):
            PytorchWheel.url_template = \
                self.config.get("agent.package_manager.torch_url_template", None)
        self.resolve_algorithm = str(
            ENV_PACKAGE_PYTORCH_RESOLVE.get() or
            self.config.get("agent.package_manager.pytorch_resolve", "pip")).lower()
        if self.resolve_algorithm not in self.resolver_types:
            print("WARNING: agent.package_manager.pytorch_resolve=={} not in {} reverting to '{}'".format(
                self.resolve_algorithm, self.resolver_types, self.resolver_types[0]))
            self.resolve_algorithm = self.resolver_types[0]

    def _init_python_ver_cuda_ver(self):
        if self.cuda is None:
            self.cuda = "cuda{}".format(self.cuda_version).lower()
        if self.python_version_string is None:
            self.python_version_string = str(self.config["agent.default_python"])
        if self.python_major_minor_str is None:
            self.python_major_minor_str = '.'.join(self.python_version_string.split('.')[:2])
            if '.' not in self.python_major_minor_str:
                raise PytorchResolutionError(
                    "invalid python version {!r} defined in configuration file, key 'agent.default_python': "
                    "must have both major and minor parts of the version (for example: '3.7')".format(
                        self.python_version_string
                    )
                )
        if self.python is None:
            self.python = "python{}".format(self.python_major_minor_str)

        if not self.exceptions:
            self.exceptions = [
                PytorchResolutionError(message)
                for message in (
                    None,
                    'cuda version "{}" is not supported'.format(self.cuda),
                    'python version "{}" is not supported'.format(
                        self.python_version_string
                    ),
                )
            ]

    @property
    def is_conda(self):
        return self.package_manager == "conda"

    @property
    def is_pip(self):
        return not self.is_conda

    def validate_python_version(self):
        """
        Make sure python version has both major and minor versions as required for choosing pytorch wheel
        """
        self._init_python_ver_cuda_ver()

        if self.is_pip and not self.python_major_minor_str:
            raise PytorchResolutionError(
                "invalid python version {!r} defined in configuration file, key 'agent.default_python': "
                "must have both major and minor parts of the version (for example: '3.7')".format(
                    self.python_version_string
                )
            )

    def match(self, req):
        if self.resolve_algorithm == "none":
            # skipping resolver
            return False

        return req.name in self.packages

    @staticmethod
    def get_platform():
        if sys.platform == "linux":
            return "linux"
        if sys.platform == "win32" or sys.platform == "cygwin":
            return "windows"
        if sys.platform == "darwin":
            return "macos"
        raise RuntimeError("unrecognized OS")

    @staticmethod
    def get_arch():
        return str(platform.machine()).lower()

    def _get_link_from_torch_page(self, req, torch_url):
        links_parser = LinksHTMLParser()
        links_parser.feed(requests.get(torch_url, timeout=10).text)
        platform_wheel = "win" if self.get_platform() == "windows" else self.get_platform()
        arch_wheel = self.get_arch()
        py_ver = self.python_major_minor_str.replace('.', '')
        url = None
        last_v = None
        closest_v = None
        # search for our package
        for l in links_parser.links:
            parts = l.split('/')[-1].split('-')
            if len(parts) < 5:
                continue
            if parts[0] != req.name:
                continue
            # version (ignore +cpu +cu92 etc. + is %2B in the file link)
            # version ignore .postX suffix (treat as regular version)
            # noinspection PyBroadException
            try:
                v = str(parts[1].split('%')[0].split('+')[0])
            except Exception:
                continue
            if len(parts) < 3 or not parts[2].endswith(py_ver):
                continue
            if len(parts) < 5 or platform_wheel not in parts[4].lower():
                continue
            if len(parts) < 5 or arch_wheel not in parts[4].lower():
                continue

            # yes this is for linux python 2.7 support, this is the only python 2.7 we support...
            if py_ver and py_ver[0] == '2' and len(parts) > 3 and not parts[3].endswith('u'):
                continue

            # check if this an actual match
            if not req.compare_version(v) or \
                    (last_v and SimpleVersion.compare_versions(last_v, '>', v, ignore_sub_versions=False)):
                continue

            # update the closest matched version (from above)
            if not closest_v:
                closest_v = v
            elif SimpleVersion.compare_versions(
                    version_a=closest_v, op='>=', version_b=v, num_parts=3) and \
                    SimpleVersion.compare_versions(
                        version_a=v, op='>=', version_b=req.specs[0][1], num_parts=3):
                closest_v = v

            url = '/'.join(torch_url.split('/')[:-1] + l.split('/'))
            last_v = v
            # if we found an exact match, use it
            # noinspection PyBroadException
            try:
                if req.specs[0][0] == '==' and \
                        SimpleVersion.compare_versions(req.specs[0][1], '==', v, ignore_sub_versions=False):
                    break
            except Exception:
                pass

        return url, last_v or closest_v

    def get_url_for_platform(self, req):
        # check if package is already installed with system packages
        self.validate_python_version()
        # noinspection PyBroadException
        try:
            if self.config.get("agent.package_manager.system_site_packages", None):
                from pip._internal.commands.show import search_packages_info
                installed_torch = list(search_packages_info([req.name]))
                # notice the comparison order, the first part will make sure we have a valid installed package
                installed_torch_version = \
                    (getattr(installed_torch[0], 'version', None) or
                     installed_torch[0]['version']) if installed_torch else None

                if installed_torch and installed_torch_version and \
                        req.compare_version(installed_torch_version):
                    print('PyTorch: requested "{}" version {}, using pre-installed version {}'.format(
                        req.name, req.specs[0] if req.specs else 'unspecified', installed_torch_version))
                    # package already installed, do nothing
                    req.specs = [('==', str(installed_torch_version))]
                    return '{} {} {}'.format(req.name, req.specs[0][0], req.specs[0][1]), True

        except Exception:
            pass

        # make sure we have a specific version to retrieve
        if not req.specs:
            req.specs = [('>', '0')]

        # noinspection PyBroadException
        try:
            req.specs[0] = (req.specs[0][0], req.specs[0][1].split('+')[0])
        except Exception:
            pass
        op, version = req.specs[0]
        # assert op == "=="

        torch_url, torch_url_key = SimplePytorchRequirement.get_torch_page(self.cuda_version)
        url, closest_matched_version = self._get_link_from_torch_page(req, torch_url)
        if not url and self.config.get("agent.package_manager.torch_nightly", None):
            torch_url, torch_url_key = SimplePytorchRequirement.get_torch_page(self.cuda_version, nightly=True)
            url, closest_matched_version = self._get_link_from_torch_page(req, torch_url)
        # try one more time, with a lower cuda version (never fallback to CPU):
        while not url and torch_url_key > 0:
            previous_cuda_key = torch_url_key
            print('Warning, could not locate PyTorch {} matching CUDA version {}, best candidate {}\n'.format(
                    req, previous_cuda_key, closest_matched_version))
            url, closest_matched_version = self._get_link_from_torch_page(req, torch_url)
            if url:
                break
            torch_url, torch_url_key = SimplePytorchRequirement.get_torch_page(int(torch_url_key)-1)
            # never fallback to CPU
            if torch_url_key < 1:
                print(
                    'Error! Could not locate PyTorch version {} matching CUDA version {}'.format(
                        req, previous_cuda_key))
                raise ValueError(
                    'Could not locate PyTorch version {} matching CUDA version {}'.format(req, self.cuda_version))
            else:
                print('Trying PyTorch CUDA version {} support'.format(torch_url_key))

        # fix broken pytorch setuptools incompatibility
        if req.name == "torch" and closest_matched_version and \
                SimpleVersion.compare_versions(closest_matched_version, "<", "1.11.0"):
            self._fix_setuptools = "setuptools < 59"

        if not url:
            url = PytorchWheel(
                torch_version=fix_version(version),
                python=self.python_major_minor_str.replace('.', ''),
                os_name=self.os,
                cuda_version=self.cuda_version,
            ).make_url()
        if url:
            # normalize url (sometimes we will get ../ which we should not...
            url = '/'.join(url.split('/')[:3]) + urllib.parse.quote(str(furl(url).path.normalize()))
            # print found
            print('Found PyTorch version {} matching CUDA version {}'.format(req, torch_url_key))

        self.log.debug("checking url: %s", url)
        return url, requests.head(url, timeout=10).ok

    @staticmethod
    def match_version(req, options):
        versioned_options = sorted(
            ((fix_version(key), value) for key, value in options.items()),
            key=itemgetter(0),
            reverse=True,
        )
        req.specs = [(op, fix_version(version)) for op, version in req.specs]

        try:
            return next(
                replacement
                for version, replacement in versioned_options
                if req.compare_version(version)
            )
        except StopIteration:
            raise PytorchResolutionError(
                'Could not find wheel for "{}", '
                "Available versions: {}".format(req, list(options))
            )

    def replace_conda(self, req):
        spec = "".join(req.specs[0]) if req.specs else ""
        if not self.cuda_version:
            return "pytorch-cpu{spec}\ntorchvision-cpu".format(spec=spec)
        return "pytorch{spec}\ntorchvision\ncuda{self.cuda_version}".format(
            self=self, spec=spec
        )

    def _table_lookup(self, req):
        """
        Look for pytorch wheel matching `req` in table
        :param req: python requirement
        """
        def check(base_, key_, exception_):
            result = base_.get(key_)
            if not result:
                if key_.startswith('cuda'):
                    print('Could not locate, {}'.format(exception_))
                    ver = sorted([float(a.replace('cuda', '').replace('none', '0')) for a in base_.keys()], reverse=True)[0]
                    key_ = 'cuda'+str(int(ver))
                    result = base_.get(key_)
                    print('Reverting to \"{}\"'.format(key_))
                    if not result:
                        raise exception_
                    return result
                raise exception_
            if isinstance(result, Exception):
                raise result
            return result

        if self.is_conda:
            return self.replace_conda(req)

        base = self.MAP
        for key, exception in zip((self.os, self.cuda, self.python), self.exceptions):
            base = check(base, key, exception)

        return self.match_version(req, base).replace(" ", "\n")

    def replace(self, req):
        # we first try to resolve things ourselves because pytorch pip is not always picking the correct
        # versions from their pip repository

        resolve_algorithm = self.resolve_algorithm
        if resolve_algorithm == "none":
            # skipping resolver
            return None
        elif resolve_algorithm == "direct":
            # noinspection PyBroadException
            try:
                new_req = self._replace(req)
                if new_req:
                    self._original_req.append((req, new_req))
                return new_req
            except Exception:
                print("Warning: Failed resolving using `pytorch_resolve=direct` reverting to `pytorch_resolve=pip`")
        elif resolve_algorithm not in self.resolver_types:
            print("Warning: `agent.package_manager.pytorch_resolve={}` "
                  "unrecognized, default to `pip`".format(resolve_algorithm))

        # check if package is already installed with system packages
        self.validate_python_version()

        # try to check if we can just use the new index URL, if we do not we will revert to old method
        try:
            extra_index_url = self.get_torch_index_url(self.cuda_version)
            if extra_index_url:
                # check if the torch version cannot be above 1.11 , we need to fix setup tools
                try:
                    if req.name == "torch" and not compare_version_rules(req.specs, [(">=", "1.11.0")]):
                        self._fix_setuptools = "setuptools < 59"
                except Exception:  # noqa
                    pass
                # now we just need to add the correct extra index url for the cuda version
                self.set_add_install_extra_index(extra_index_url[0])

                if req.specs and len(req.specs) == 1 and req.specs[0][0] == "==":
                    # remove any +cu extension and let pip resolve that
                    # and add .* if we have 3 parts version to deal with nvidia container 'a' version
                    # i.e. "1.13.0" -> "1.13.0.*" so it should match preinstalled "1.13.0a0+936e930"
                    spec_3_parts = req.format_specs(num_parts=3)
                    spec_max3_parts = req.format_specs(max_num_parts=3)
                    if spec_3_parts == spec_max3_parts and not spec_max3_parts.endswith("*"):
                        line = "{} {}.*".format(req.name, spec_max3_parts)
                    else:
                        line = "{} {}".format(req.name, spec_max3_parts)

                    if req.marker:
                        line += " ; {}".format(req.marker)
                else:
                    # return the original line
                    line = req.line

                print("PyTorch: Adding index `{}` and installing `{}`".format(extra_index_url[0], line))

                return line

        except Exception:  # noqa
            pass

        try:
            new_req = self._replace(req)
            if new_req:
                self._original_req.append((req, new_req))
            return new_req
        except Exception as e:
            message = "Exception when trying to resolve python wheel"
            self.log.debug(message, exc_info=True)
            raise PytorchResolutionError("{}: {}".format(message, e))

    def _replace(self, req):
        self.validate_python_version()
        try:
            result, ok = self.get_url_for_platform(req)
            self.log.debug('Replacing requirement "%s" with %r', req, result)
            return result
        except:
            pass

        # try:
        #     result = self._table_lookup(req)
        # except Exception as e:
        #     exc = e
        # else:
        #     self.log.debug('Replacing requirement "%s" with %r', req, result)
        #     return result
        # self.log.debug(
        #     "Could not find Pytorch wheel in table, trying manually constructing URL"
        # )

        result = ok = None
        # try:
        #     result, ok = self.get_url_for_platform(req)
        # except Exception:
        #     pass

        if not ok:
            if result:
                self.log.debug("URL not found: {}".format(result))
            exc = PytorchResolutionError(
                "Could not find pytorch wheel URL for: {} with cuda {} support".format(req, self.cuda_version)
            )
            # cancel exception chaining
            six.raise_from(exc, None)

        self.log.debug('Replacing requirement "%s" with %r', req, result)
        return result

    def replace_back(self, list_of_requirements):  # type: (Dict) -> Dict
        """
        :param list_of_requirements: {'pip': ['a==1.0', ]}
        :return: {'pip': ['a==1.0', ]}
        """
        def build_specific_version_req(a_line, a_name, a_new_req):
            try:
                r = Requirement.parse(a_line)
                wheel_parts = r.uri.split("/")[-1].split('-')
                version = str(wheel_parts[1].split('%')[0].split('+')[0])
                new_r = Requirement.parse("{} == {} # {}".format(a_name, version, str(a_new_req)))
                if new_r.line:
                    # great it worked!
                    return new_r.line
            except:  # noqa
                pass
            return None

        if not self._original_req:
            return list_of_requirements
        try:
            for k, lines in list_of_requirements.items():
                # k is either pip/conda
                if k not in ('pip', 'conda'):
                    continue
                for i, line in enumerate(lines):
                    if not line or line.lstrip().startswith('#'):
                        continue
                    parts = [p for p in re.split(r'\s|=|\.|<|>|~|!|@|#', line) if p]
                    if not parts:
                        continue
                    for req, new_req in self._original_req:
                        if req.req.name == parts[0]:
                            # support for pip >= 20.1
                            if '@' in line:
                                # skip if we have nothing to add
                                if str(req).strip() != str(new_req).strip():
                                    # if this is local file and use the version detection
                                    if req.local_file:
                                        lines[i] = '{}'.format(str(new_req))
                                    else:
                                        # try to rebuild requirements with specific version:
                                        new_line = build_specific_version_req(line, req.req.name, new_req)
                                        if new_line:
                                            lines[i] = new_line
                                        else:
                                            lines[i] = '{} # {}'.format(str(req), str(new_req))
                            else:
                                new_line = build_specific_version_req(line, req.req.name, new_req)
                                if new_line:
                                    lines[i] = new_line
                                else:
                                    lines[i] = '{} # {}'.format(line, str(new_req))
                            break
        except:
            pass

        return list_of_requirements

    def post_scan_add_req(self):  # type: () -> Optional[MarkerRequirement]
        """
        Allows the RequirementSubstitution to add an extra line/requirements after
        the initial requirements scan is completed.
        Called only once per requirements.txt object
        """
        if self._fix_setuptools:
            return MarkerRequirement(Requirement.parse(self._fix_setuptools))
        return None

    def get_torch_index_url(self, cuda_version, nightly=False):
        # noinspection PyBroadException
        try:
            cuda = int(cuda_version)
        except Exception:
            cuda = 0

        if nightly:
            for c in range(cuda, max(-1, cuda-15), -1):
                # then try the nightly builds, it might be there...
                torch_url = self.nightly_extra_index_url_template.format(c)
                # noinspection PyBroadException
                try:
                    if requests.get(torch_url, timeout=10).ok:
                        print('Torch nightly CUDA {} index page found'.format(c))
                        self.torch_index_url_lookup[c] = torch_url
                        return self.torch_index_url_lookup[c], c
                except Exception:
                    pass
            return

        # first check if key is valid
        if cuda in self.torch_index_url_lookup:
            return self.torch_index_url_lookup[cuda], cuda

        # then try a new cuda version page
        for c in range(cuda, max(-1, cuda-15), -1):
            torch_url = self.extra_index_url_template.format(c)
            # noinspection PyBroadException
            try:
                if requests.get(torch_url, timeout=10).ok:
                    print('Torch CUDA {} index page found, adding `{}`'.format(c, torch_url))
                    self.torch_index_url_lookup[c] = torch_url
                    return self.torch_index_url_lookup[c], c
            except Exception:
                pass

        keys = sorted(self.torch_index_url_lookup.keys(), reverse=True)
        for k in keys:
            if k <= cuda:
                return self.torch_index_url_lookup[k], k
        # return default - zero
        return self.torch_index_url_lookup[0], 0

    MAP = {
        "windows": {
            "cuda100": {
                "python3.7": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl"
                },
                "python3.6": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl"
                },
                "python3.5": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-win_amd64.whl"
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda92": {
                "python3.7": {
                    "0.4.1",
                    "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl",
                },
                "python3.6": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl"
                },
                "python3.5": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl"
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda91": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-win_amd64.whl"
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-win_amd64.whl"
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda90": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl",
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda80": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-win_amd64.whl",
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cudanone": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-win_amd64.whl",
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
        },
        "macos": {
            "cuda100": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda92": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda91": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda90": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda80": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cudanone": {
                "python3.6": {"0.4.0": "torch"},
                "python3.5": {"0.4.0": "torch"},
                "python2.7": {"0.4.0": "torch"},
            },
        },
        "linux": {
            "cuda100": {
                "python3.7": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp37-cp37m-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp37-cp37m-manylinux1_x86_64.whl",
                },
                "python3.6": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl",
                },
                "python3.5": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp35-cp35m-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp35-cp35m-manylinux1_x86_64.whl",
                },
                "python2.7": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp27-cp27mu-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp27-cp27mu-manylinux1_x86_64.whl",
                },
            },
            "cuda92": {
                "python3.7": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp37-cp37m-manylinux1_x86_64.whl"
                },
                "python3.6": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp36-cp36m-manylinux1_x86_64.whl"
                },
                "python3.5": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp35-cp35m-manylinux1_x86_64.whl"
                },
                "python2.7": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp27-cp27mu-manylinux1_x86_64.whl"
                },
            },
            "cuda91": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl"
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl"
                },
                "python2.7": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl"
                },
            },
            "cuda90": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                },
                "python2.7": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                },
            },
            "cuda80": {
                "python3.6": {
                    "0.4.1": "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl",
                    "0.3.1": "torch==0.3.1",
                    "0.3.0.post4": "torch==0.3.0.post4",
                    "0.1.2.post1": "torch==0.1.2.post1",
                    "0.1.2": "torch==0.1.2",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                },
                "python3.5": {
                    "0.4.1": "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl",
                    "0.3.1": "torch==0.3.1",
                    "0.3.0.post4": "torch==0.3.0.post4",
                    "0.1.2.post1": "torch==0.1.2.post1",
                    "0.1.2": "torch==0.1.2",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                },
                "python2.7": {
                    "0.4.1": "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl",
                    "0.3.1": "torch==0.3.1",
                    "0.3.0.post4": "torch==0.3.0.post4",
                    "0.1.2.post1": "torch==0.1.2.post1",
                    "0.1.2": "torch==0.1.2",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                },
            },
            "cudanone": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                },
                "python2.7": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                },
            },
        },
    }

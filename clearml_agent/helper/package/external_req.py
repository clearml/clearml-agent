import re
from collections import OrderedDict
from typing import Text

from ..._vendor.pathlib2 import Path

from .base import PackageManager
from .requirements import SimpleSubstitution
from ..base import safe_furl as furl


class ExternalRequirements(SimpleSubstitution):

    name = "external_link"
    cwd = None

    def __init__(self, *args, **kwargs):
        super(ExternalRequirements, self).__init__(*args, **kwargs)
        self.post_install_req = []
        self.post_install_req_lookup = OrderedDict()
        self.post_install_local_req_lookup = OrderedDict()

    def match(self, req):
        # match local folder building:
        if self.is_local_folder_package(req):
            # noinspection PyBroadException
            try:
                folder_path = req.req.line.strip().split('#')[0].strip()
                if self.cwd and not Path(folder_path).is_absolute():
                    folder_path = (Path(self.cwd) / Path(folder_path)).absolute().as_posix()
                self.post_install_local_req_lookup['file://{}'.format(folder_path)] = req.req.line
            except Exception:
                pass
            return True

        # match both editable or code or unparsed
        if not (not req.name or req.req and (req.req.editable or req.req.vcs)):
            return False
        if not req.req or not req.req.line or not req.req.line.strip() or req.req.line.strip().startswith('#'):
            return False
        if req.pip_new_version and not (req.req.editable or req.req.vcs):
            return False
        return True

    def post_install(self, session):
        post_install_req = self.post_install_req
        self.post_install_req = []
        for req in post_install_req:
            if self.is_already_installed(req):
                print("No need to reinstall \'{}\' from VCS, "
                      "the exact same version is already installed".format(req.name))
                continue

            if not req.pip_new_version:
                # noinspection PyBroadException
                try:
                    freeze_base = PackageManager.out_of_scope_freeze() or dict(pip=[])
                except Exception:
                    freeze_base = dict(pip=[])

            req_line = self._add_vcs_credentials(req, session)

            # if we have older pip version we have to make sure we replace back the package name with the
            # git repository link. In new versions this is supported and we get "package @ git+https://..."
            if not req.pip_new_version:
                PackageManager.out_of_scope_install_package(req_line, "--no-deps")
                # noinspection PyBroadException
                try:
                    freeze_post = PackageManager.out_of_scope_freeze() or dict(pip=[])
                    package_name = list(set(freeze_post['pip']) - set(freeze_base['pip']))
                    if package_name and package_name[0] not in self.post_install_req_lookup:
                        self.post_install_req_lookup[package_name[0]] = req.req.line
                except Exception:
                    pass

            # no need to force reinstall, pip will always rebuild if the package comes from git
            # and make sure the required packages are installed (if they are not it will install them)
            if not PackageManager.out_of_scope_install_package(req_line):
                raise ValueError("Failed installing GIT/HTTPs package \'{}\'".format(req_line))

    @staticmethod
    def _add_vcs_credentials(req, session):
        req_line = req.tostr(markers=False)
        if req_line.strip().startswith('-e ') or req_line.strip().startswith('--editable'):
            req_line = re.sub(r'^(-e|--editable=?)\s*', '', req_line, count=1)
        if req.req.vcs and req_line.startswith('git+'):
            try:
                url_no_frag = furl(req_line)
                url_no_frag.set(fragment=None)
                # reverse replace
                fragment = req_line[::-1].replace(url_no_frag.url[::-1], '', 1)[::-1]
                vcs_url = req_line[4:]
                # reverse replace
                vcs_url = vcs_url[::-1].replace(fragment[::-1], '', 1)[::-1]
                # notice git:// is actually ssh://
                if vcs_url and vcs_url.startswith('git://'):
                    vcs_url = vcs_url.replace('git://', 'ssh://', 1)

                from ..repo import Git
                vcs = Git(session=session, url=vcs_url, location=None, revision=None)
                vcs._set_ssh_url()
                new_req_line = 'git+{}{}'.format(vcs.url_with_auth, fragment)
                if new_req_line != req_line:
                    furl_line = furl(new_req_line)
                    print('Replacing original pip vcs \'{}\' with \'{}\''.format(
                        req_line,
                        furl_line.set(password='xxxxxx').tostr() if furl_line.password else new_req_line))
                    req_line = new_req_line
            except Exception:
                print('WARNING: Failed parsing pip git install, using original line {}'.format(req_line))
        return req_line

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Store in post req install, and return nothing
        self.post_install_req.append(req)
        # mark skip package, we will install it in post install hook
        return Text('')

    def replace_back(self, list_of_requirements):
        if not list_of_requirements:
            return list_of_requirements

        for k in list_of_requirements:
            # k is either pip/conda
            if k not in ('pip', 'conda'):
                continue
                
            original_requirements = list_of_requirements[k]
            list_of_requirements[k] = [r for r in original_requirements
                                       if r not in self.post_install_req_lookup]
            list_of_requirements[k] += [self.post_install_req_lookup.get(r, '')
                                        for r in self.post_install_req_lookup.keys() if r in original_requirements]

            if self.post_install_local_req_lookup:
                original_requirements = list_of_requirements[k]
                list_of_requirements[k] = [
                    r for r in original_requirements
                    if len(r.split('@', 1)) != 2 or r.split('@', 1)[1].strip() not in self.post_install_local_req_lookup]

                list_of_requirements[k] += [
                    self.post_install_local_req_lookup.get(r.split('@', 1)[1].strip(), '')
                    for r in original_requirements
                    if len(r.split('@', 1)) == 2 and r.split('@', 1)[1].strip() in self.post_install_local_req_lookup]

        return list_of_requirements

    @classmethod
    def is_local_folder_package(cls, req):
        # noinspection PyBroadException
        try:
            if not req.name and req.req and not req.req.editable and not req.req.vcs and \
                    req.req.line and req.req.line.strip().split('#')[0] and \
                    not req.req.line.strip().split('#')[0].lower().endswith('.whl') and \
                    not (req.req.line.strip().startswith('-r ') or req.req.line.strip().startswith('--requirement ')):
                return True
        except Exception:
            pass
        return False


class OnlyExternalRequirements(ExternalRequirements):
    def __init__(self, *args, **kwargs):
        super(OnlyExternalRequirements, self).__init__(*args, **kwargs)

    def match(self, req):
        return True

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Do not store the skipped requirements
        # mark skip package
        if super(OnlyExternalRequirements, self).match(req):
            if self.is_already_installed(req):
                print("No need to reinstall \'{}\' from VCS, "
                      "the exact same version is already installed".format(req.name))
                return Text('')

            return self._add_vcs_credentials(req, self._session)

        return Text('')

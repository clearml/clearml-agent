import re
from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class PriorityPackageRequirement(SimpleSubstitution):

    name = ("cython", "numpy", "setuptools", "pip", "uv", )
    optional_package_names = tuple()

    def __init__(self, *args, **kwargs):
        super(PriorityPackageRequirement, self).__init__(*args, **kwargs)
        self._replaced_packages = {}
        # check if we need to replace the packages:
        priority_packages = self.config.get('agent.package_manager.priority_packages', None)
        if priority_packages:
            self.__class__.name = [p.lower() for p in priority_packages]
        priority_optional_packages = self.config.get('agent.package_manager.priority_optional_packages', None)
        if priority_optional_packages:
            self.__class__.optional_package_names = [p.lower() for p in priority_optional_packages]

    def match(self, req):
        # match both Cython & cython
        return req.name and (req.name.lower() in self.name or req.name.lower() in self.optional_package_names)

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        self._replaced_packages[req.name] = req.line

        if req.name.lower() in self.optional_package_names:
            # noinspection PyBroadException
            try:
                if PackageManager.out_of_scope_install_package(str(req)):
                    return Text(req)
            except Exception:
                pass
            return Text('')
        PackageManager.out_of_scope_install_package(str(req))
        return Text(req)

    def replace_back(self, list_of_requirements):
        """
        :param list_of_requirements: {'pip': ['a==1.0', ]}
        :return: {'pip': ['a==1.0', ]}
        """
        # if we replaced setuptools, it means someone requested it, and since freeze will not contain it,
        # we need to add it manually
        if not self._replaced_packages:
            return list_of_requirements

        # we assume that both pip & setup tools are not in list_of_requirements, and we need to add them

        if "pip" in self._replaced_packages:
            full_freeze = PackageManager.out_of_scope_freeze(freeze_full_environment=True)
            if not full_freeze:
                if "pip" in list_of_requirements:
                    list_of_requirements["pip"] = [self._replaced_packages["pip"]] + list_of_requirements["pip"]
            else:
                # now let's look for pip
                pips = [line for line in full_freeze.get("pip", []) if str(line.split("==")[0]).strip() == "pip"]
                if pips and "pip" in list_of_requirements:
                    list_of_requirements["pip"] = [pips[0]] + list_of_requirements["pip"]

        if "setuptools" in self._replaced_packages:
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
                        # if we found setuptools, do nothing
                        if parts[0] == "setuptools":
                            return list_of_requirements

                # if we are here it means we have not found setuptools
                # we should add it:
                if "pip" in list_of_requirements:
                    list_of_requirements["pip"] = [self._replaced_packages["setuptools"]] + list_of_requirements["pip"]

            except Exception as ex:  # noqa
                return list_of_requirements

        return list_of_requirements


class CachedPackageRequirement(PriorityPackageRequirement):

    name = ("setuptools", "pip", )
    optional_package_names = tuple()

    def replace(self, req):
        """
        Put the requirement in the list for later conversion
        :raises: ValueError if version is pre-release
        """
        self._replaced_packages[req.name] = req.line
        return Text(req)


class PackageCollectorRequirement(SimpleSubstitution):
    """
    This RequirementSubstitution class will allow you to have multiple instances of the same
    package, it will output the last one (by order) to be actually used.
    """
    name = tuple()

    def __init__(self, session, collect_package):
        super(PackageCollectorRequirement, self).__init__(session)
        self._collect_packages = collect_package or tuple()
        self._last_req = None

    def match(self, req):
        # match package names
        return req.name and req.name.lower() in self._collect_packages

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        self._last_req = req.clone()
        return ''

    def post_scan_add_req(self):
        """
        Allows the RequirementSubstitution to add an extra line/requirements after
        the initial requirements scan is completed.
        Called only once per requirements.txt object
        """
        last_req = self._last_req
        self._last_req = None
        return last_req

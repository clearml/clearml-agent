import json
import re
import shlex
from copy import copy

from clearml_agent.backend_api.session import Request
from clearml_agent.helper.docker_args import DockerArgsSanitizer
from clearml_agent.helper.package.requirements import (
    RequirementsManager, MarkerRequirement,
    compare_version_rules, )


def resolve_default_container(session, task_id, container_config, ignore_match_rules=False):
    container_lookup = session.config.get('agent.default_docker.match_rules', None)
    if not session.check_min_api_version("2.13") or not container_lookup:
        return container_config, None

    # check backend support before sending any more requests (because they will fail and crash the Task)
    try:
        session.verify_feature_set('advanced')
    except ValueError:
        # ignoring matching rules only supported in higher tiers
        return container_config, None

    if ignore_match_rules:
        print("INFO: default docker command line override, ignoring default docker container match rules")
        # ignoring matching rules only supported in higher tiers
        return container_config, None

    result = session.send_request(
        service='tasks',
        action='get_all',
        version='2.14',
        json={'id': [task_id],
              'only_fields': ['script.requirements', 'script.binary',
                              'script.repository', 'script.branch',
                              'project', 'container', 'tags', 'user'],
              'search_hidden': True},
        method=Request.def_method,
        async_enable=False,
    )
    try:
        task_info = result.json()['data']['tasks'][0] if result.ok else {}
    except (ValueError, TypeError):
        return container_config, None

    from clearml_agent.external.requirements_parser.requirement import Requirement

    # store tasks repository
    repository = task_info.get('script', {}).get('repository') or ''
    branch = task_info.get('script', {}).get('branch') or ''
    binary = task_info.get('script', {}).get('binary') or ''
    requested_container = task_info.get('container', {})

    # get project full path
    project_full_name = ''
    if task_info.get('project', None):
        result = session.send_request(
            service='projects',
            action='get_all',
            version='2.13',
            json={
                'id': [task_info.get('project')],
                'only_fields': ['name'],
            },
            method=Request.def_method,
            async_enable=False,
        )
        try:
            if result.ok:
                project_full_name = result.json()['data']['projects'][0]['name'] or ''
        except (ValueError, TypeError):
            pass

    match_term_lookup = {
        "project": project_full_name,
        "project_id": task_info.get('project', ''),
        "script.repository": repository,
        "script.branch": branch,
        "script.binary": binary,
        "user_id": task_info.get('user', ""),
        "container": requested_container.get('image', ''),
        "tags": task_info.get('tags', []),
    }

    task_packages_lookup = {}
    for entry in container_lookup:
        match = entry.get('match', None)
        if not match:
            continue

        matched = True
        for key, value in match_term_lookup.items():
            term = match.get(key, None)
            if not term:
                continue
            values = [value] if not isinstance(value, (list, tuple)) else value
            # noinspection PyBroadException
            try:
                terms = [term] if not isinstance(term, (list, tuple)) else term
                # we fail if we didn't find ANY match in the list
                if all(any(bool(re.search(t, v)) for v in values) for t in terms):
                    # we found a match, go to the next match term
                    pass
                else:
                    # no match, stop, and we should go to the next rule
                    matched = False
                    break
            except Exception:
                print('Failed parsing regular expression \"{}\" in rule: {}'.format(term, entry))
                matched = False
                break

        # we had at least a single key that was Not matched in this rule, go to the next one
        if not matched:
            continue

        # look for the complicated stuff (i.e. requirements)
        for req_section in ['script.requirements.pip', 'script.requirements.conda']:
            if not match.get(req_section, None):
                continue

            match_pip_reqs = [MarkerRequirement(Requirement.parse('{} {}'.format(k, v)))
                              for k, v in match.get(req_section, None).items()]

            if not task_packages_lookup.get(req_section):
                req_section_parts = req_section.split('.')
                task_packages_lookup[req_section] = \
                    RequirementsManager.parse_requirements_section_to_marker_requirements(
                        requirements=task_info.get(req_section_parts[0], {}).get(
                            req_section_parts[1], {}).get(req_section_parts[2], None)
                    )

            matched_all_reqs = True
            for mr in match_pip_reqs:
                matched_req = False
                for pr in task_packages_lookup[req_section]:
                    if mr.req.name != pr.req.name:
                        continue
                    if compare_version_rules(mr.specs, pr.specs):
                        matched_req = True
                        break
                if not matched_req:
                    matched_all_reqs = False
                    break

            # if ew have a match, check second section
            if matched_all_reqs:
                continue
            # no match stop
            matched = False
            break

        # if we found a match in the rulebook
        if matched:
            allow_override = bool(entry.get('force_container_rules', None))
            container_config["force_container_rules"] = allow_override

            if not container_config.get('image') or allow_override:
                container_config['image'] = entry.get('image', None) or ''

            if not container_config.get('arguments') or allow_override:
                container_config['arguments'] = entry.get('arguments', None) or ''

            if not container_config.get('setup_shell_script') or allow_override:
                container_config['setup_shell_script'] = entry.get('setup_shell_script', None) or ''
                if isinstance(container_config['setup_shell_script'], (list, tuple)):
                    container_config['setup_shell_script'] = "\n".join(container_config['setup_shell_script'])

            update_back_task = entry.get('update_back_task', None)
            if update_back_task is None:
                update_back_task = session.config.get('agent.default_docker.update_back_task', None)

            container_config['update_back_task'] = update_back_task

            return container_config, entry

    return container_config, None


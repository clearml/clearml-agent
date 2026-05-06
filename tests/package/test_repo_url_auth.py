import pytest

from clearml_agent.helper.repo import Git

NO_CHANGE = object()


def param(url, expected, user=False, password=False):
    """
    Helper function for creating parametrization arguments.
    :param url: input url
    :param expected: expected output URL or NO_CHANGE if the same as input URL
    :param user: Add `agent.git_user=USER` to config
    :param password: Add `agent.git_password=PASSWORD` to config
    """
    expected_repr = "NO_CHANGE" if expected is NO_CHANGE else None
    user = "USER" if user else None
    password = "PASSWORD" if password else None
    return pytest.param(
        url,
        expected,
        user,
        password,
        id="-".join(filter(None, (url, user, password, expected_repr))),
    )


@pytest.mark.parametrize(
    "url,expected,user,password",
    [
        param("https://bitbucket.org/company/repo", NO_CHANGE),
        param("https://bitbucket.org/company/repo", NO_CHANGE, user=True),
        param("https://bitbucket.org/company/repo", NO_CHANGE, password=True),
        param(
            "https://bitbucket.org/company/repo", NO_CHANGE, user=True, password=True
        ),
        param("https://user@bitbucket.org/company/repo", NO_CHANGE),
        param("https://user@bitbucket.org/company/repo", NO_CHANGE, user=True),
        param("https://user@bitbucket.org/company/repo", NO_CHANGE, password=True),
        param(
            "https://user@bitbucket.org/company/repo",
            "https://USER:PASSWORD@bitbucket.org/company/repo",
            user=True,
            password=True,
        ),
        param("https://user:password@bitbucket.org/company/repo", NO_CHANGE),
        param("https://user:password@bitbucket.org/company/repo", NO_CHANGE, user=True),
        param(
            "https://user:password@bitbucket.org/company/repo", NO_CHANGE, password=True
        ),
        param(
            "https://user:password@bitbucket.org/company/repo",
            NO_CHANGE,
            user=True,
            password=True,
        ),
        param("ssh://git@bitbucket.org/company/repo", NO_CHANGE),
        param("ssh://git@bitbucket.org/company/repo", NO_CHANGE, user=True),
        param("ssh://git@bitbucket.org/company/repo", NO_CHANGE, password=True),
        param(
            "ssh://git@bitbucket.org/company/repo", NO_CHANGE, user=True, password=True
        ),
        param("git@bitbucket.org:company/repo.git", NO_CHANGE),
        param("git@bitbucket.org:company/repo.git", NO_CHANGE, user=True),
        param("git@bitbucket.org:company/repo.git", NO_CHANGE, password=True),
        param(
            "git@bitbucket.org:company/repo.git", NO_CHANGE, user=True, password=True
        ),
    ],
)


# TODO: add tests for conversion:
# git+https://bitbucket.org/seematics/trains-internal.git@master#subdirectory=.
# git+ssh://git@bitbucket.org/seematics/trains-internal.git@main#subdirectory=a
# git+git://git@bitbucket.org/seematics/trains-internal.git@master#subdirectory=b
#
# git+https://github.com/clearml/clearml.git@master#subdirectory=.
# git+ssh://git@github.com/clearml/clearml.git@main#subdirectory=a
# git+git://git@github.com/clearml/clearml.git@master#subdirectory=b


def test(url, user, password, expected):
    config = {"agent": {"git_user": user, "git_pass": password}}
    result = Git.add_auth(config, url)
    expected = result if expected is NO_CHANGE else expected
    assert result == expected

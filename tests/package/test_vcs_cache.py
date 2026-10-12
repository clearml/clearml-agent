import logging
import os
import subprocess
import time

import pytest
from pathlib2 import Path

from clearml_agent.external.pyhocon import ConfigFactory
from clearml_agent.helper.base import ExecutionInfo
from clearml_agent.helper.hash import md5
from clearml_agent.helper.os.locks import FileLock
from clearml_agent.helper.repo import (
    cleanup_vcs_cache_orphans,
    clone_repository_cached,
    DEFAULT_VCS_CACHE_ORPHAN_MIN_AGE_SEC,
)


DEFAULT_AGENT_CONF = Path(__file__).resolve().parents[2] / "clearml_agent/backend_api/config/default/agent.conf"


class FakeSession(object):
    debug_mode = False

    def __init__(self, vcs_cache_path=None, **vcs_cache):
        self.config = ConfigFactory.from_dict({"agent": ConfigFactory.parse_file(str(DEFAULT_AGENT_CONF))})
        if vcs_cache_path:
            vcs_cache["path"] = str(vcs_cache_path)
        for k, v in vcs_cache.items():
            self.config.put("agent.vcs_cache.{}".format(k), v)

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)


def test_cleanup_vcs_cache_orphans(tmp_path):
    repo_hash = md5(b"x").hexdigest()
    old = time.time() - DEFAULT_VCS_CACHE_ORPHAN_MIN_AGE_SEC - 60
    orphan_old = tmp_path / "repo.{}_0123456789".format(repo_hash)
    orphan_old_exp = tmp_path / "repo.{}_06e-05".format(repo_hash)
    orphan_new = tmp_path / "repo.{}_0987654321".format(repo_hash)
    valid = tmp_path / "repo.{}".format(repo_hash)
    for d in (orphan_old, orphan_old_exp, orphan_new, valid):
        (d / "repo").mkdir(parents=True)
    lock_file = tmp_path / "{}.lock".format(repo_hash)
    lock_file.write_text(u"")
    for p in (orphan_old, orphan_old_exp, valid, lock_file):
        os.utime(str(p), (old, old))

    assert cleanup_vcs_cache_orphans(FakeSession(tmp_path)) == 2
    assert not orphan_old.exists()
    assert not orphan_old_exp.exists()
    # too new, might still be in use by an older agent
    assert orphan_new.exists()
    assert valid.exists()
    assert lock_file.exists()


def test_cleanup_vcs_cache_orphans_disabled(tmp_path):
    orphan = tmp_path / "repo.{}_0123456789".format(md5(b"x").hexdigest())
    orphan.mkdir()
    old = time.time() - DEFAULT_VCS_CACHE_ORPHAN_MIN_AGE_SEC - 60
    os.utime(str(orphan), (old, old))
    assert cleanup_vcs_cache_orphans(FakeSession(tmp_path, cleanup_orphans=False)) == 0
    assert orphan.exists()


def test_cleanup_vcs_cache_orphans_min_age_from_config(tmp_path):
    orphan = tmp_path / "repo.{}_0123456789".format(md5(b"x").hexdigest())
    orphan.mkdir()
    old = time.time() - 2 * 60 * 60
    os.utime(str(orphan), (old, old))
    assert cleanup_vcs_cache_orphans(FakeSession(tmp_path)) == 0
    assert cleanup_vcs_cache_orphans(FakeSession(tmp_path, orphan_min_age_sec=60 * 60)) == 1
    assert not orphan.exists()


def test_cleanup_vcs_cache_orphans_missing_folder(tmp_path):
    assert cleanup_vcs_cache_orphans(FakeSession(tmp_path / "missing")) == 0


def _git(*args, **kwargs):
    subprocess.check_call(("git",) + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)


@pytest.fixture
def remote_repo(tmp_path, monkeypatch):
    # the agent runs "git config --global ...", keep it away from the real user config
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    src = tmp_path / "src"
    src.mkdir()
    _git("init", "-q", "-b", "master", str(src))
    (src / "a.txt").write_text(u"hello")
    _git("add", "a.txt", cwd=str(src))
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=str(src))
    bare = tmp_path / "remote" / "repo.git"
    _git("clone", "-q", "--bare", str(src), str(bare))
    return bare.as_uri()


def _clone(session, url, destination):
    return clone_repository_cached(session, ExecutionInfo(repository=url, branch="master"), destination)


def test_clone_uses_single_cache_folder(tmp_path, remote_repo):
    cache = tmp_path / "vcs-cache"
    session = FakeSession(cache)
    for i in range(2):
        dest = tmp_path / "work{}".format(i)
        _clone(session, remote_repo, dest)
        assert (dest / "repo.git" / "a.txt").read_text() == u"hello"
    assert len([p for p in cache.iterdir() if p.is_dir()]) == 1


def test_clone_lock_timeout_bypasses_cache(tmp_path, remote_repo):
    cache = tmp_path / "vcs-cache"
    cache.mkdir()
    session = FakeSession(cache, lock_timeout_sec=0.5)
    repo_hash = md5(remote_repo.encode()).hexdigest()
    other_agent_lock = FileLock(filename=(cache / "{}.lock".format(repo_hash)).as_posix())
    other_agent_lock.acquire()
    try:
        dest = tmp_path / "work"
        _clone(session, remote_repo, dest)
    finally:
        other_agent_lock.release()
    assert (dest / "repo.git" / "a.txt").read_text() == u"hello"
    # nothing was added to the vcs cache except the lock file
    assert [p.name for p in cache.iterdir()] == ["{}.lock".format(repo_hash)]


def test_clone_lock_timeout_temp_cache_legacy(tmp_path, remote_repo):
    cache = tmp_path / "vcs-cache"
    cache.mkdir()
    session = FakeSession(cache, lock_timeout_sec=0.5, temp_cache_on_lock_timeout=True)
    repo_hash = md5(remote_repo.encode()).hexdigest()
    other_agent_lock = FileLock(filename=(cache / "{}.lock".format(repo_hash)).as_posix())
    other_agent_lock.acquire()
    try:
        dest = tmp_path / "work"
        _clone(session, remote_repo, dest)
    finally:
        other_agent_lock.release()
    assert (dest / "repo.git" / "a.txt").read_text() == u"hello"
    # the temporary cache folder was removed, only the (other agent's) lock file is left
    assert [p.name for p in cache.iterdir()] == ["{}.lock".format(repo_hash)]


def test_clone_lock_timeout_temp_cache_legacy_removed_on_failure(tmp_path, remote_repo):
    cache = tmp_path / "vcs-cache"
    cache.mkdir()
    session = FakeSession(cache, lock_timeout_sec=0.5, temp_cache_on_lock_timeout=True)
    repo_hash = md5(remote_repo.encode()).hexdigest()
    other_agent_lock = FileLock(filename=(cache / "{}.lock".format(repo_hash)).as_posix())
    other_agent_lock.acquire()
    try:
        with pytest.raises(Exception):
            # checkout of a missing branch fails after the temp cache folder was cloned
            clone_repository_cached(
                session, ExecutionInfo(repository=remote_repo, branch="missing-branch"), tmp_path / "work")
    finally:
        other_agent_lock.release()
    assert [p.name for p in cache.iterdir()] == ["{}.lock".format(repo_hash)]

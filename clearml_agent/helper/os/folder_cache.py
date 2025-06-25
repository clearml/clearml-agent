import os
import shutil
from logging import warning
from random import random
from time import time
from typing import List, Optional, Sequence

import psutil
from ..._vendor.pathlib2 import Path

from .locks import FileLock


class FolderCache(object):
    _lock_filename = '.clearml.lock'
    _def_lock_timeout_seconds = 30
    _temp_entry_prefix = '_temp.'

    def __init__(self, cache_folder, max_cache_entries=5, min_free_space_gb=None, lock_timeout_seconds=None):
        self._cache_folder = Path(os.path.expandvars(cache_folder)).expanduser().absolute()
        self._cache_folder.mkdir(parents=True, exist_ok=True)
        self._max_cache_entries = max_cache_entries
        self._last_copied_entry_folder = None
        self._min_free_space_gb = min_free_space_gb if min_free_space_gb and min_free_space_gb > 0 else None
        self._lock = FileLock((self._cache_folder / self._lock_filename).as_posix())
        self._lock_timeout_seconds = float(lock_timeout_seconds or self._def_lock_timeout_seconds)

    def get_cache_folder(self):
        # type: () -> Path
        """
        :return: Return the base cache folder
        """
        return self._cache_folder

    def copy_cached_entry(self, keys, destination):
        # type: (List[str], Path) -> Optional[Path]
        """
        Copy a cached entry into a destination directory, if the cached entry does not exist return None
        :param keys:
        :param destination:
        :return: Target path, None if cached entry does not exist
        """
        self._last_copied_entry_folder = None
        if not keys:
            return None

        # lock so we make sure no one deletes it before we copy it
        # noinspection PyBroadException
        try:
            self._lock.acquire(timeout=self._lock_timeout_seconds, readonly=True)
        except BaseException as ex:
            warning('Could not lock cache folder {}: {}'.format(self._cache_folder, ex))
            import traceback
            warning('DEBUG: Exception {}: {}'.format(ex, traceback.format_exc()))
            return None

        src = None
        try:
            src = self.get_entry(keys)
            if src:
                destination = Path(destination).absolute()
                destination.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(destination.as_posix())
                shutil.copytree(src.as_posix(), dst=destination.as_posix(), symlinks=True)
        except BaseException as ex:
            warning('Could not copy cache folder {} to {}: {}'.format(src, destination, ex))
            self._lock.release()
            return None

        # release Lock
        self._lock.release()

        self._last_copied_entry_folder = src
        return destination if src else None

    def get_entry(self, keys):
        # type: (List[str]) -> Optional[Path]
        """
        Return a folder (a sub-folder of inside the cache_folder) matching one of the keys
        :param keys: List of keys, return the first match to one of the keys, notice keys cannot contain '.'
        :return: Path to the sub-folder or None if none was found
        """
        if not keys:
            return None
        # conform keys
        keys = [keys] if isinstance(keys, str) else keys
        keys = sorted([k.replace('.', '_') for k in keys])
        for cache_folder in self._cache_folder.glob('*'):
            if cache_folder.is_dir() and any(True for k in cache_folder.name.split('.') if k in keys):
                cache_folder.touch()
                return cache_folder
        return None

    def add_entry(self, keys, source_folder, exclude_sub_folders=None):
        # type: (List[str], Path, Optional[Sequence[str]]) -> bool
        """
        Add a local folder into the cache, copy all sub-folders inside `source_folder`
        excluding folders matching `exclude_sub_folders` list
        :param keys: Cache entry keys list (str)
        :param source_folder: Folder to copy into the cache
        :param exclude_sub_folders: List of sub-folders to exclude from the copy operation
        :return: return True is new entry was added to cache
        """
        if not keys:
            return False

        keys = [keys] if isinstance(keys, str) else keys
        keys = sorted([k.replace('.', '_') for k in keys])

        # If entry already exists skip it
        cached_entry = self.get_entry(keys)
        if cached_entry:
            # make sure the entry contains all keys
            cached_keys = cached_entry.name.split('.')
            if set(keys) - set(cached_keys):
                # noinspection PyBroadException
                try:
                    self._lock.acquire(timeout=self._lock_timeout_seconds)
                except BaseException as ex:
                    warning('Could not lock cache folder {}: {}'.format(self._cache_folder, ex))
                    import traceback
                    warning('DEBUG: Exception {}: {}'.format(ex, traceback.format_exc()))
                    # failed locking do nothing
                    return True
                keys = sorted(list(set(keys) | set(cached_keys)))
                dst = cached_entry.parent / '.'.join(keys)
                # rename
                try:
                    shutil.move(src=cached_entry.as_posix(), dst=dst.as_posix())
                except BaseException as ex:
                    warning('Could not rename cache entry {} to {}: ex'.format(
                        cached_entry.as_posix(), dst.as_posix(), ex))
                # release lock
                self._lock.release()
            return True

        # make sure we remove old entries
        self._remove_old_entries()

        # if we do not have enough free space, do nothing.
        if not self._check_min_free_space():
            warning('Could not add cache entry, not enough free space on drive, '
                    'free space threshold {} GB. Clearing all cache entries!'.format(self._min_free_space_gb))
            self._remove_old_entries(max_cache_entries=0)
            return False

        # create the new entry for us
        exclude_sub_folders = exclude_sub_folders or []
        source_folder = Path(source_folder).absolute()
        # create temp folder
        temp_folder = \
            self._temp_entry_prefix + \
            '{}.{}'.format(str(time()).replace('.', '_'), str(random()).replace('.', '_'))
        temp_folder = self._cache_folder / temp_folder
        temp_folder.mkdir(parents=True, exist_ok=False)

        for f in source_folder.glob('*'):
            if f.name in exclude_sub_folders:
                continue
            if f.is_dir():
                shutil.copytree(
                    src=f.as_posix(), dst=(temp_folder / f.name).as_posix(),
                    symlinks=True, ignore_dangling_symlinks=True)
            else:
                shutil.copy(
                    src=f.as_posix(), dst=(temp_folder / f.name).as_posix(),
                    follow_symlinks=False)

        # rename the target folder
        target_cache_folder = self._cache_folder / '.'.join(keys)
        # if we failed moving it means someone else created the cached entry before us, we can just leave
        # noinspection PyBroadException
        try:
            shutil.move(src=temp_folder.as_posix(), dst=target_cache_folder.as_posix())
        except BaseException:
            # noinspection PyBroadException
            try:
                shutil.rmtree(path=temp_folder.as_posix())
            except BaseException:
                return False

        return True

    def get_last_copied_entry(self):
        # type: () -> Optional[Path]
        """
        :return: the last copied cached entry folder inside the cache
        """
        return self._last_copied_entry_folder

    def _remove_old_entries(self, max_cache_entries=None):
        # type: (Optional[int]) -> ()
        """
        Notice we only keep self._max_cache_entries-1, assuming we will be adding a new entry soon
        :param int max_cache_entries: if not None use instead of self._max_cache_entries
        """
        folder_entries = [(cache_folder, cache_folder.stat().st_mtime)
                          for cache_folder in self._cache_folder.glob('*')
                          if cache_folder.is_dir() and not cache_folder.name.startswith(self._temp_entry_prefix)]
        folder_entries = sorted(folder_entries, key=lambda x: x[1], reverse=True)

        number_of_entries_to_keep = self._max_cache_entries - 1 \
            if max_cache_entries is None else max(0, int(max_cache_entries))

        # if nothing to do, leave
        if not folder_entries[number_of_entries_to_keep:]:
            return

        # lock so we make sure no one deletes it before we copy it
        # noinspection PyBroadException
        try:
            self._lock.acquire(timeout=self._lock_timeout_seconds)
        except BaseException as ex:
            warning('Could not lock cache folder {}: {}'.format(self._cache_folder, ex))
            import traceback
            warning('DEBUG: Exception {}: {}'.format(ex, traceback.format_exc()))
            return

        for folder, ts in folder_entries[number_of_entries_to_keep:]:
            try:
                shutil.rmtree(folder.as_posix(), ignore_errors=True)
            except BaseException as ex:
                warning('Could not delete cache entry {}: {}'.format(folder.as_posix(), ex))

        self._lock.release()

    def _check_min_free_space(self):
        # type: () -> bool
        """
        :return: return False if we hit the free space limit.
        If not free space limit provided, always return True
        """
        if not self._min_free_space_gb or not self._cache_folder:
            return True
        free_space = float(psutil.disk_usage(self._cache_folder.as_posix()).free)
        free_space /= 2**30
        return free_space > self._min_free_space_gb

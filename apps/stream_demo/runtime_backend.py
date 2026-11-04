"""Keep Windows/WSL launchers separate from the self-contained Linux runtime."""
import os


def native_runtime():
    return os.environ.get('STRAW_RUNTIME') == 'native'


def host_command(distro, *args):
    return list(args) if native_runtime() else ['wsl.exe', '-d', distro, '-u', 'root', '--', *args]

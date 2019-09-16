import argparse
from subprocess import call
import os

parser = argparse.ArgumentParser(description='Script for required components for rok.')
parser.add_argument('--download-firedrake-install-script', nargs='?', const=True, type=bool)
parsed, unknown_parsed_args = parser.parse_known_args()
# print(parsed.download_firedrake_install_script)


def get_args_to_firedrake_install(parser, unknown_args):

    for arg in unknown_args:
        if arg.startswith("--"):
            parser.add_argument(arg, action="store_true")

    options = [str(option).replace('_', '-') for option in unknown_args]
    return str(" ".join(options))


def download_firedrake():
    call(
        "curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install",
        shell=True
    )


def install_firedrake(install_args):
    call(
        f"python3 firedrake-install {install_args}", shell=True
    )


if parsed.download_firedrake_install_script:
    download_firedrake()

args_to_firedrake_install = get_args_to_firedrake_install(parser, unknown_parsed_args)
install_firedrake(args_to_firedrake_install)

# args_to_firedrake_install = get_args_to_firedrake_install()
# root_directory_path = os.getcwd()



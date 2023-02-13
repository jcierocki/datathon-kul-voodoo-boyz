#!/bin/bash

PYTHON_VERSION=3.10.10

if [ -z "$1" ]
then
	echo "You need to provide path to destination you want a new venv to be created an argument while running this script."
	exit 1
fi

if [ -d "$1" ]
then
	read -p "The catalog you want to create venv in already exists. Do you want to proceed? (y/n) " yn
	case $yn in
		y ) echo ok, we will proceed;;
		n ) echo exiting...;
			exit;;
		* ) echo invalid response;
			exit 1;;
	esac
fi

if ! command -v pyenv &> /dev/null
then
	echo -e 'Install pyenv first. See https://github.com/pyenv/pyenv. I personally recommend using manual "Basic GitHub Checkout" method instead of "Automatic Installer" due to higher relability.'
	exit 1
fi

{
	pyenv update &> /dev/null &&
	echo "Automatic pyenv installer detected."
} || {
	pyenv_path=$(which pyenv | grep -o ".*/\.pyenv")
        echo -e "Standard pyenv installation detected in $pyenv_path."
	git -C $pyenv_path pull
}

pyenv install -s $PYTHON_VERSION
pyenv local $PYTHON_VERSION

python -m venv $1
source "$1/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

exit 0

import os
from func import *
from dotenv import load_dotenv, find_dotenv


load_dotenv()
USER = os.environ['USER_NAME']
ROOT = os.environ['ROOT_PATH']


if __name__ == '__main__':

    os.chdir(ROOT)

    task = input("clone / push: ")

    if task == 'clone':
        clone_git(USER)

    elif task == 'push':
        push_git()
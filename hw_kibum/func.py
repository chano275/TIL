import sys
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def clone_git(user):
    print(f"gitlab ID: {user}")
    subject = input("Subject: ")
    if not os.path.exists(subject):
        os.makedirs(subject)

    os.chdir(subject)
    current_folder = os.getcwd()

    print(f"clone repositories in {current_folder}...")

    seperators = ['hw', 'ws']
    set_num = int(input("Set number: "))

    for sep in seperators:
        if sep == 'hw':
            stage = [2, 4]
        else:
            stage = [1, 2, 3, 4, 5, 'a', 'b', 'c']

        for st in stage:
            if os.path.exists(f"{subject}_{sep}_{set_num}_{st}"):
                print(f"{subject}_{sep}_{set_num}_{st} already exists.")
                os.chdir(f"{subject}_{sep}_{set_num}_{st}")
                subprocess.run(['git', 'pull', 'origin', 'master'])
                os.chdir('..')
            else:
                url = f"https://lab.ssafy.com/{user}/{subject}_{sep}_{set_num}_{st}"
                print(f"git clone {url}")
                subprocess.run(['git', 'clone', url])


def push_git():  # USER_ID 사용되지 않음 
    print("Select one")
    push_task = input("[1.push] / 2.commit / 3.add / 4.only backup: ")
    push_task = 1 if not push_task else int(push_task)
    subject = input("Subject: ")
    set_num = input("Set number: ")
    backup_dir = input("Backup directory absolute path(without git).If you want to skip it, press Enter: ")

    if push_task <= 2:
        print("Type the commit massage. It will come after the directory name. (ex: {dir_name} <your_commit_massage>)\n:")
        commit_message = input("Commit massage: ")
    os.chdir(subject)
    dir_list = os.listdir()

    print("Start task...")
    if backup_dir:
        print(f"Backup files to {backup_dir}")
    for i, dir in enumerate(dir_list):
        *_, sep, set_num_dir, st = dir.split('_')
        if set_num_dir == set_num:
            os.chdir(dir)
            if push_task <= 3:
                subprocess.run(['git', 'add', '.'])
            if push_task <= 2:
                subprocess.run(['git', 'commit', '-m', f'{dir} {commit_message}'])
            if push_task <= 1:
                subprocess.run(['git', 'push', 'origin', 'master'])
            
            if backup_dir:
                abs_path = os.path.abspath('.')
                if not os.path.exists(subject):
                    subprocess.run(['cp', '-r', abs_path, backup_dir])
                    subprocess.run(['rm', '-rf', os.path.join(backup_dir, dir, '.git')])

            os.chdir('..')
        progressBar(i,len(dir_list))
import os, subprocess

task = input("clone / push:")

if task == 'clone':
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
                url = f"https://lab.ssafy.com/chano01794/{subject}_{sep}_{set_num}_{st}"
                print(f"git clone {url}")
                subprocess.run(['git', 'clone', url])

elif task == 'push':
    subject = input("Subject: ")
    os.chdir(subject)

    dir_list = os.listdir()

    for dir in dir_list:
        os.chdir(dir)
        try:
            subprocess.run(['git', 'add', '.'])
            subprocess.run(['git', 'commit', '-m', f'{dir} 제출'])
            subprocess.run(['git', 'push', 'origin', 'master'])
        except: 
            pass
        os.chdir('..')

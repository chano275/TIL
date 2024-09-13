# GitAuto

**SSYFY 교육생**을 위한 자동 gitlab pull / push 툴입니다.

## Get Stared

1. 해당 repository를 fork후 clone 혹은 clone합니다.
```bash
$ git clone https://github.com/qja1998/GitAuto.git
```
1-1. dotenv 가 설치되어 있지 않다면 설치합니다.

`pip install python-dotenv`


2. .env 파일을 설정합니다.

`.env.tamplate` -> `.env`로 파일명을 변경한 후 자신에 맞는 정보로 수정합니다.
`USER`에 자신의 gitlab 아이디를 입력하고, `ROOT_PATH`에 자신이 gitlab repository를 저장할 폴더를 입력합니다.
```
# Your gitlab id
USER = YOUR_GITLAB_ID

# Root directory you want
ROOT_PATH = YOUR_ROOT_DIR
```


3. `git_auto.py`를 실행합니다.

```bash
clone / push:
```
원하는 작업을 선택합니다.


- Clone task

```bash
Subject:
```
clone 하고싶은 과목을 입력합니다. (python, web, ect...)


```bash
Set number:
```
clone 하려는 set number를 입력합니다.


이후 clone이 진행됩니다.

- Push task

```bash
Select one
[1.push] / 2.commit / 3.add / 4.only backup: 
```

원하는 작업을 선택합니다.(1, 2, 3, 4)

default는 push이며 입력하지 않고 enter를 입력할 경우 push가 선택됩니다.

push를 선택하면 push, commit, add, backup을 모두 수행합니다.

commit를 선택하면 commit, add, backup을 모두 수행하며. add를 선택하면 add, backup을 모두 수행하는 방식입니다.

백업만 하고 싶다면 4번을 선택하면 됩니다.

```bash
Subject:
Set number:
```

마찬가지로 과목과 set number를 입력합니다.

```bash
Backup directory absolute path(without git).If you want to skip it, press Enter: 
```

backup하고 싶은 target directory를 지정합니다. 절대경로를 입력해주면 되며, `.git`을 제외하고 복사됩니다.

만약 backup을 원하지 않으면 아무것도 입력하지 않으면 됩니다.

```bash
Type the commit massage. It will come after the directory name. (ex: {dir_name} <your_commit_massage>)
:
```

commit이 포함된 작업일 경우 commit massage를 입력합니다. 기본적으로 repository_name + commit_massage 형태입니다.

이후 입력에 따라 작업이 진행됩니다.

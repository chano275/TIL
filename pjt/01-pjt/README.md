# 01-pjt

- problem a : 
pprint(response)  /  print(type(response)) # 와 같은 명령어들을 통해 데이터 내부를 보는걸 우선으로, 
우리가 필요한 요소에 접근하는 법을 먼저 생각하는데 중요하다는 것을 알았다. 
타입을 찍어보는게 구조 이해하기 훨씬 수월해질 수 있어 시간단축이 더더욱 잘 될듯하다. 
당연한 얘기지만, 데이터의 구조와 수업과 달라 api와 데이터를 파악하는데 각 문제마다 시간이 꽤나 걸렸다. 
기본적으로 dict 형태로 받아와져 있는데, results 라는 key 안에 생각했던 데이터가 존재했다.
이후 pjt01 테이블 생성 및 생성한 csv 파일 통해서 데이터 삽입 완료 

- problem b : 
a 에서 저장한 csv 파일에서 id 읽어와 리스트에 저장 
api마다의 상세 정보를 구해야 하는 문제였어서, for문을 넓게 돌려 str_id 만큼 반복했다. 
다른 문제들과 다르게 response 에서 item 을 보는게 아니라, fields를 loop돌리면서 원하는 정보를 얻는 식으로 문제를 풀었다.
'장르'에 담겨있는 데이터가 문자열이 아니라, dict 형태로 저장되어 있었는데, 
해당 객체의 내부 loop를 통해서 문제에서 원하는 , 로 이어지는 문자열을 구할 수 있었다. 
개인적으로 아쉬웠던 부분은, 장르 문자열을 만듦에 있어서 공간을 조금 많이 쓴 것 같아
다음에 수정할 수 있을 때에 이 부분을 만져보고 싶다. 

- problem c :
a와 동일한 방식으로 다시 돌아왔고, break 해야 할, 요소를 추가하면 안되는 case 들을 생각해야 해서
break와 함께 이용할 flag 변수를 선언해 control 하는 식으로 프로그램을 짰다.
기억에 남는 점은 rating 요소가 다른 fields 와 동일한 곳에 있는게 아니라, author_details 안에 존재했던게 기억에 남았다. 
None 에 대한 처리가 생각보다 까다로웠고 이후 d를 짜면서 든 생각은 if else 문에 대한 처리를 b에서의 문자열 처리와 동일하게
시간날때 수정해보고 싶다는 생각을 했다. 

- problem d : 
마지막의 마지막에 가서야 코드를 짬에 있어서 큰그림을 잘못 그렸다는걸 확인할 수 있었다.
primary key 인 cast_id 에 중복이 있어서 csv file 을 넣는데 에러가 발생했고
primary key와 스키마에 대한 확인이 우선시 되어야 코드를 짬에 있어서 조건 체크를 확실히 할 수 있다는 것을 상기하게 되었다.
추가적으로 item[key] 에 대한 접근이 읽는것만이 아니라, 수정도 가능한게 아닐까...? 하는 마음에 코드를 짜 보았지만
테스트시에는 잘 되었는데, 확실하게 체크하지는 못해서 아쉬움이 있었다.

최종적으로 완성을 하지 못한 코드였고, b ~ d 의 각각의 부분에 아쉬운 부분들이 있었지만 많은 배움을 얻을 수 있는 시간이었다.




## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://lab.ssafy.com/chano01794/01-pjt.git
git branch -M master
git push -uf origin master
```

## Integrate with your tools

- [ ] [Set up project integrations](https://lab.ssafy.com/chano01794/01-pjt/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

# -*- coding: utf-8 -*-
from random import choice

names = ["吴可鑫","王蕊","魏华鹏","董澍威","陈东","陈永杰","任海瑞","孔笑宇","贾云兵","周梓俊"]
Distributed=[i for i in range(59)]

all_students = []

for i in range(10):
	student_j = []
	for j in range(6):
		if len(Distributed) != 0:
			num_q = choice(Distributed)
			student_j.append(num_q)
			Distributed.remove(num_q)
		else:
			break
	student_j.sort()
	all_students.append(student_j)

# print("已分配情况：", all_students)
# print("未分配题目", Distributed)

common_contects = []
path = "C:\\Users\\A\\Desktop\\common.txt"
with open(path, 'r') as f:
	common_contects = f.readlines()

all_s_ori = []
for s in all_students:
	student_ori = []
	for q_i in s:
		question = common_contects[q_i]
		student_ori.append('Q'+question.strip())
	all_s_ori.append(student_ori)

undisc_list = []
if len(Distributed) == 0:
	pass
else:
	for un_disc in Distributed:
		un_q = common_contects[un_disc]
		undisc_list.append(un_q)

print("已分配情况：")
for name, item in zip(names, all_s_ori):
	print(name, '=>', item)
# print()
# print("未分配题目", undisc_list)

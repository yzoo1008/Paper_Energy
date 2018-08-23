import numpy as np
import openpyxl
import os

local = [[108, "서울"], [159, "부산"], [176, "대구"], [112, "인천"], [156, "광주"], [133, "대전"]]
data = dict()
for element in local:
	data[element[0]] = []

for year in range(2014, 2016):
	wb = openpyxl.load_workbook('./기후/' + str(year) + '.xlsx', data_only=True)
	sheet = wb.worksheets[0]
	row_c = sheet.max_row
	col_c = sheet.max_column
	for row in range(2, row_c + 1):
		key = sheet[row][0].value
		frag = []
		for col in range(2, 8):
			frag.append(sheet[row][col].value)
			data[key].append(frag)

array = []
for key in data.keys():
	array.append(data[key])

if not "data" in os.listdir("./"):
	os.mkdir("data")

np.save("./data/data.npy", array)
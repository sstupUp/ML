year = int(input("년도를 입력하세요:"))
is_leap = False

if (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0)):
    is_leap = True

if is_leap:
    print(year, "는 윤년입니다")
else:
    print(year, "는 윤년이 아닙니다")
year = int(input("년도를 입력하세요: "))
month = int(input("월을 입력하세요: "))
day = int(input("일을 입력하세요: "))

is_leap = False

if (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0)):
    is_leap = True

day_of_year = (month - 1) * 31 + day

if month >= 3 and month <=12:
    day_of_year = day_of_year - ((4*month + day) // 10)

    if is_leap:
        day_of_year = day_of_year + 1

print("통일: ", day_of_year)
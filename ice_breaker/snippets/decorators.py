import time
# import myPackage
# from myPackage import greet
# from myPackage.greet import hello

from myPackage import *




def wraper(func):
    def modified(a):
        start_time = time.time()
        func(a)
        print(f'Total execution time: {time.time() - start_time}')

    return modified


@wraper
def add_range_of_num(num):
    x = 0
    for num in range(num):
       x += num

    return x

add_range_of_num(1000000)


# print(myPackage.greet.hello())
# print(greet.hello())
# print(hello())

print(hello())


__all__ = ['wraper']
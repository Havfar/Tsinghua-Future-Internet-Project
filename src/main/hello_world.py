import sys
print(sys.path)

f = open("hello_world.txt","w+")
f.write("Hello World!")
f.close()

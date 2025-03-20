FC = gfortran
CC = gcc
AR = ar
CFLAGS = -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk -I/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/c++/v1 
MODULES = -I/Users/zeno/Documents/COLLIER-1.2.8/modules
COLLIER = -L. -lcollier
FFLAGS = -O2 -Wall 
LDFLAGS = -L/usr/local/Cellar/gcc/9.2.0/lib/gcc/9/ -lgfortran 

bridge.o: bridge.f90
	gfortran -c bridge.f90 $(COLLIER) -o bridge.o $(MODULES) 

libbridge.a: bridge.o
	$(AR) rcs libbridge.a bridge.o

cpp_wrapper.cpp: libbridge.a
	gcc $(CFLAGS) cpp_wrapper.cpp -L. -lbridge $(COLLIER) -o cpp_wrapper_ex -lc++ $(LDFLAGS) 

cpp_wrapper.o: 
	gcc $(CFLAGS) -c cpp_wrapper.cpp

all: cpp_wrapper.o libbridge.a
	gcc -shared $(CFLAGS) -o cpp_wrapper.so cpp_wrapper.o -L. -lbridge $(COLLIER) -lc++ $(LDFLAGS) 

clean:
	rm libbridge.a *.o
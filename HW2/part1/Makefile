CXX := clang++
ifeq (/usr/bin/clang++-11,$(wildcard /usr/bin/clang++-11*))
    CXX=clang++-11
endif
CXX := clang++
ifeq (/usr/bin/clang++-16,$(wildcard /usr/bin/clang++-16*))
    CXX=clang++-16
endif
CXXFLAGS := -O3 -std=c++17 -Wall  -mavx2 -mfma

all: pi.out

pi.o: pi.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -c $< -o $@

pi.out: pi.o
	$(CXX) $(CXXFLAGS) -I./include $< -o $@ -lpthread

clean:
	$(RM) *.o *.s pi.out *~

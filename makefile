all: bin/main

bin/main:obj/bitmap.o obj/algebra.o obj/main.o obj/NNUtils.o

	g++ -o bin/main -g -lm -Wall -pedantic  -Iinclude obj/bitmap.o obj/main.o obj/algebra.o \
	obj/NNUtils.o


obj/bitmap.o:include/bitmap.h src/bitmap.cpp

	g++ -o obj/bitmap.o  -g -lm -Wall -pedantic -Iinclude -c src/bitmap.cpp


obj/algebra.o:include/algebra.h src/algebra.cpp

	g++ -o obj/algebra.o  -g -lm -Wall -pedantic -Iinclude -c src/algebra.cpp


obj/NNUtils.o:include/NNUtils.h src/NNUtils.cpp

	g++ -o obj/NNUtils.o  -g -lm -Wall -pedantic -Iinclude -c src/NNUtils.cpp


obj/main.o:src/main.cpp

	g++ -o obj/main.o -g -lm -Wall -pedantic  -Iinclude -c src/main.cpp


clean:

	-rm bin/*


mrproper:clean

	-rm obj/* data/*.txt data/*.bmp

clima:

	-rm ./*.bmp
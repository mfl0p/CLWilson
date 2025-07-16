CC = g++
LD = $(CC)

.SUFFIXES:
.SUFFIXES: .o .c .h .cl .cpp

VERSION_MAJOR := 0
VERSION_MINOR := 3
date := $(shell powershell.exe get-date -format FileDate)

APP = CLWilson-win64-v$(VERSION_MAJOR).$(VERSION_MINOR)-$(date).exe

SRC = main.cpp cl_wilson.cpp cl_wilson.h simpleCL.c simpleCL.h kernels/clearn.cl kernels/clearresult.cl kernels/iterate.cl kernels/setup.cl kernels/getsegprps.cl kernels/mulsmall.cl kernels/mullarge.cl kernels/reduce.cl kernels/find.cl kernels/common.cl putil.c putil.h
KERNEL_HEADERS = kernels/clearn.h kernels/clearresult.h kernels/iterate.h kernels/setup.h kernels/getsegprps.h kernels/mulsmall.h kernels/mullarge.h kernels/reduce.h kernels/find.h kernels/common.h
OBJ = main.o cl_wilson.o simpleCL.o putil.o

LIBS = OpenCL.dll

BOINC_DIR = C:/mingwbuilds/boinc
BOINC_INC = -I$(BOINC_DIR)/lib -I$(BOINC_DIR)/api -I$(BOINC_DIR) -I$(BOINC_DIR)/win_build
BOINC_LIB = -L$(BOINC_DIR)/lib -L$(BOINC_DIR)/api -L$(BOINC_DIR) -lboinc_opencl -lboinc_api -lboinc

CFLAGS  = -I . -I kernels -O3 -m64 -Wall -DVERSION_MAJOR=\"$(VERSION_MAJOR)\" -DVERSION_MINOR=\"$(VERSION_MINOR)\"
LDFLAGS = $(CFLAGS) -lstdc++ -static

all : clean $(APP)

$(APP) : $(OBJ)
	$(LD) $(LDFLAGS) $^ $(LIBS) $(BOINC_LIB) -o $@ libprimesievewin.a libgmpwin.a

main.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ main.cpp

cl_wilson.o : $(SRC) $(KERNEL_HEADERS)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ cl_wilson.cpp

simpleCL.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ simpleCL.c

putil.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ putil.c

.cl.h:
	perl cltoh.pl $< > $@

clean :
	del *.o
	del kernels\*.h
	del $(APP)


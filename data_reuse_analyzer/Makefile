SOURCE_FILES	=	\
			Main.cpp OptionsProcessor.cpp ConfigProcessor.cpp Utility.cpp

BINARY_FILE	=	polyscientist

BARVINOK_INSTALL = /nfs_home/stavarag/work/software/barvinok/barvinok-0.41.2_install
PET_INSTALL = /nfs_home/stavarag/work/software/barvinok/barvinok-0.41.2_install
NTL_INSTALL = /nfs_home/stavarag/work/software/ntl-11.3.2_install
ISL_INSTALL = /nfs_home/stavarag/work/software/barvinok/isl_install

###########################################################################
## You should not need to change anything below this line.               ##
###########################################################################

CXX = g++
LIBRARY_FLAGS = -L$(BARVINOK_INSTALL)/lib -L$(ISL_INSTALL)/lib -L$(PET_INSTALL)/lib -L$(NTL_INSTALL)/lib -lpet -lisl -lbarvinok  -lntl  -lgmp  -lpolylibgmp -lpthread
CXXFLAGS = -O2 -I$(BARVINOK_INSTALL)/include -I$(PET_INSTALL)/include -I$(ISL_INSTALL)/include -I . 

TEMP0_FILES = $(SOURCE_FILES:.cpp=.o)
TEMP1_FILES = $(TEMP0_FILES:.C=.o)
OBJECT_FILES = $(TEMP1_FILES:.cc=.o)

all		:	$(BINARY_FILE)

$(BINARY_FILE)	:	$(OBJECT_FILES)
			$(CXX) -o $(BINARY_FILE) $(LDFLAGS) $(OBJECT_FILES) $(LIBRARY_FLAGS)
                        
.cpp.o          :
			$(CXX) -c $(CXXFLAGS) -o $@ $<
.C.o            :
			$(CXX) -c $(CXXFLAGS) -o $@ $<
.cc.o           :
			$(CXX) -c $(CXXFLAGS) -o $@ $<

clean		:
			rm -f *.o
			rm $(BINARY_FILE)


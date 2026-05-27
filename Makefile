PETSC_DIR ?= /data/app/petsc
PETSC_ARCH ?=
PETSC_PKG ?= PETSc
PKG_CONFIG ?= pkg-config

TARGET = bin/control_arm_cpp
SRC = src/main.cpp src/geometry.cpp src/density.cpp src/vtk.cpp src/petsc_utils.cpp
OBJ = $(SRC:.cpp=.o)
PETSC_CONF = ${PETSC_DIR}/lib/petsc/conf

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS += -Iinclude

ifneq ($(wildcard ${PETSC_CONF}/variables),)
include ${PETSC_CONF}/variables
include ${PETSC_CONF}/rules

all: ${TARGET}
wsl: ${TARGET}

${TARGET}: ${OBJ} | bin
	${CLINKER} -o $@ ${OBJ} ${PETSC_KSP_LIB}

else
ifeq ($(origin CXX),default)
CXX = mpicxx
endif
PETSC_CFLAGS := $(shell ${PKG_CONFIG} --cflags ${PETSC_PKG} 2>/dev/null || ${PKG_CONFIG} --cflags petsc 2>/dev/null)
PETSC_LIBS := $(shell ${PKG_CONFIG} --libs ${PETSC_PKG} 2>/dev/null || ${PKG_CONFIG} --libs petsc 2>/dev/null)

all: wsl
wsl: ${TARGET}

${TARGET}: ${SRC} | bin
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${PETSC_CFLAGS} -o $@ ${SRC} ${PETSC_LIBS} -lm
endif

bin:
	mkdir -p bin

result:
	mkdir -p result

clean:
	rm -f src/*.o ${TARGET}
.PHONY: all wsl clean result

PETSC_DIR ?= /data/app/petsc
PETSC_ARCH ?=
PETSC_PKG ?= PETSc
PKG_CONFIG ?= pkg-config

TARGET = bin/control_arm_petsc
SRC = src/control_arm_petsc.c
OBJ = src/control_arm_petsc.o
PETSC_CONF = ${PETSC_DIR}/lib/petsc/conf

ifneq ($(wildcard ${PETSC_CONF}/variables),)
include ${PETSC_CONF}/variables
include ${PETSC_CONF}/rules

all: ${TARGET}
wsl: ${TARGET}

${TARGET}: ${OBJ} | bin
	${CLINKER} -o $@ ${OBJ} ${PETSC_KSP_LIB}

else
ifeq ($(origin CC),default)
CC = mpicc
endif
CFLAGS ?= -O3 -std=c99 -Wall -Wextra
PETSC_CFLAGS := $(shell ${PKG_CONFIG} --cflags ${PETSC_PKG} 2>/dev/null || ${PKG_CONFIG} --cflags petsc 2>/dev/null)
PETSC_LIBS := $(shell ${PKG_CONFIG} --libs ${PETSC_PKG} 2>/dev/null || ${PKG_CONFIG} --libs petsc 2>/dev/null)

all: wsl

wsl: ${TARGET}

${TARGET}: ${SRC} | bin
	${CC} ${CFLAGS} ${PETSC_CFLAGS} -o $@ $< ${PETSC_LIBS} -lm

endif

bin:
	mkdir -p bin

clean:
	rm -f src/*.o ${TARGET}

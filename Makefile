PETSC_DIR_CANDIDATE := $(firstword $(wildcard $(HOME)/petsc/petsc-v* $(HOME)/petsc /data/app/petsc))
PETSC_DIR ?= $(if $(PETSC_DIR_CANDIDATE),$(PETSC_DIR_CANDIDATE),/data/app/petsc)
PETSC_ARCH_CANDIDATE := $(firstword $(notdir $(wildcard $(PETSC_DIR)/arch-*)))
PETSC_ARCH ?= $(PETSC_ARCH_CANDIDATE)
PETSC_PKG ?= PETSc
PKG_CONFIG ?= pkg-config

PETSC_CONF = ${PETSC_DIR}/lib/petsc/conf
ifneq ($(strip $(PETSC_ARCH)),)
PETSC_INCLUDE_GUESS = -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB_GUESS = -L${PETSC_DIR}/${PETSC_ARCH}/lib -L${PETSC_DIR}/lib -lpetsc
else
PETSC_INCLUDE_GUESS = -I${PETSC_DIR}/include
PETSC_LIB_GUESS = -L${PETSC_DIR}/lib -lpetsc
endif

TARGET = bin/control_arm_cpp
SRC = src/main.cpp src/geometry.cpp src/density.cpp src/vtk.cpp src/petsc_utils.cpp src/h8_matrix_free.cpp src/density_pipeline.cpp src/low_order_optimizer.cpp src/h8_optimizer.cpp src/ann_model.cpp src/emsfem_ann.cpp
OBJ = $(SRC:.cpp=.o)

CXXSTD ?= c++11
CXXFLAGS ?= -O3 -std=$(CXXSTD) -Wall -Wextra -pedantic
CPPFLAGS += -Iinclude

ifneq ($(wildcard ${PETSC_CONF}/variables),)
include ${PETSC_CONF}/variables

all: ${TARGET}
wsl: ${TARGET}

${TARGET}: ${SRC} | bin
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${PETSC_CC_INCLUDES} -o $@ ${SRC} ${PETSC_KSP_LIB} -lm

else
ifeq ($(origin CXX),default)
CXX = mpicxx
endif
PETSC_CFLAGS := $(shell ${PKG_CONFIG} --cflags ${PETSC_PKG} 2>/dev/null || ${PKG_CONFIG} --cflags petsc 2>/dev/null)
PETSC_LIBS := $(shell ${PKG_CONFIG} --libs ${PETSC_PKG} 2>/dev/null || ${PKG_CONFIG} --libs petsc 2>/dev/null)
ifeq ($(strip $(PETSC_CFLAGS)),)
PETSC_CFLAGS := $(PETSC_INCLUDE_GUESS)
endif
ifeq ($(strip $(PETSC_LIBS)),)
PETSC_LIBS := $(PETSC_LIB_GUESS)
endif

all: wsl
wsl: ${TARGET}

${TARGET}: ${SRC} | bin
	@printf '#include <petscsys.h>\n' | ${CXX} ${CPPFLAGS} ${PETSC_CFLAGS} -x c++ -E - >/dev/null 2>&1 || \
	  (echo "PETSc headers not found by the active compiler flags." >&2; \
	   echo "Set PETSC_DIR/PETSC_ARCH or PKG_CONFIG_PATH, for example:" >&2; \
	   echo "  find /data/home/dlut_ycx /data/app -name petscsys.h 2>/dev/null | head" >&2; \
	   echo "  make all CXXSTD=c++11 PETSC_DIR=/path/to/petsc PETSC_ARCH=/path-or-arch-if-needed" >&2; \
	   exit 2)
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${PETSC_CFLAGS} -o $@ ${SRC} ${PETSC_LIBS} -lm
endif

bin:
	mkdir -p bin

result:
	mkdir -p result

clean::
	rm -f src/*.o ${TARGET}
.PHONY: all wsl clean result

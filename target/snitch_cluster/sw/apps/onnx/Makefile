MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SRC_DIR  := $(realpath $(MK_DIR)/$(APP)/src)
ONNX_SRC_DIR  := $(realpath $(MK_DIR)../../../../../sw/onnx/maxpool/src)
ONNX_DATA_DIR  := $(realpath $(MK_DIR)../../../../../sw/onnx/maxpool/data)

SRCS     = $(realpath $(SRC_DIR)/$(APP).c)
SRCS    += $(realpath $(ONNX_SRC_DIR)/$(APP).c)
SRCS    += $(realpath $(ONNX_SRC_DIR)/_$(APP).c)
SRCS    += $(realpath $(ONNX_SRC_DIR)/_$(APP)_single_core.c)
INCDIRS += $(SRC_DIR) $(ONNX_SRC_DIR) $(ONNX_DATA_DIR)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data

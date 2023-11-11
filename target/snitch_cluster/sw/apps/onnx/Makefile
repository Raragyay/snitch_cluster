MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SRC_DIR  := $(realpath $(MK_DIR)/$(APP)/src)
ONNX_DIR  := $(realpath $(MK_DIR)../../../../../sw/onnx/src)

SRCS     = $(realpath $(SRC_DIR)/$(APP).c)
SRCS    += $(realpath $(ONNX_DIR)/$(APP).c)
INCDIRS += $(SRC_DIR) $(ONNX_DIR)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data

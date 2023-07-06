# OPTION(DR "Option description" ON)
# IF(DR)
#     ADD_DEFINITIONS(-DDR)
# ENDIF(DR)

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE . ${PROJECT_SOURCE_DIR}/Common)
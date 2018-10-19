add_executable(test_collective EXCLUDE_FROM_ALL test_collective.cc)

include_directories(../)
target_link_libraries(test_collective _collective_mode ${HOOMD_LIBRARIES} ${PYTHON_LIBRARIES})

#enable_testing()
#add_test(NAME test_collective COMMAND test_collective)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS test_collective)

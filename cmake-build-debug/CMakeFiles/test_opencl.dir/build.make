# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\programs\clion\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\programs\clion\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\projects\OpenclProjects\vector_add

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\projects\OpenclProjects\vector_add\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/test_opencl.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_opencl.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_opencl.dir/flags.make

CMakeFiles/test_opencl.dir/main.cpp.obj: CMakeFiles/test_opencl.dir/flags.make
CMakeFiles/test_opencl.dir/main.cpp.obj: CMakeFiles/test_opencl.dir/includes_CXX.rsp
CMakeFiles/test_opencl.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\projects\OpenclProjects\vector_add\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_opencl.dir/main.cpp.obj"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\test_opencl.dir\main.cpp.obj -c C:\projects\OpenclProjects\vector_add\main.cpp

CMakeFiles/test_opencl.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_opencl.dir/main.cpp.i"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\projects\OpenclProjects\vector_add\main.cpp > CMakeFiles\test_opencl.dir\main.cpp.i

CMakeFiles/test_opencl.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_opencl.dir/main.cpp.s"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\projects\OpenclProjects\vector_add\main.cpp -o CMakeFiles\test_opencl.dir\main.cpp.s

# Object files for target test_opencl
test_opencl_OBJECTS = \
"CMakeFiles/test_opencl.dir/main.cpp.obj"

# External object files for target test_opencl
test_opencl_EXTERNAL_OBJECTS =

test_opencl.exe: CMakeFiles/test_opencl.dir/main.cpp.obj
test_opencl.exe: CMakeFiles/test_opencl.dir/build.make
test_opencl.exe: CMakeFiles/test_opencl.dir/linklibs.rsp
test_opencl.exe: CMakeFiles/test_opencl.dir/objects1.rsp
test_opencl.exe: CMakeFiles/test_opencl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\projects\OpenclProjects\vector_add\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_opencl.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\test_opencl.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_opencl.dir/build: test_opencl.exe

.PHONY : CMakeFiles/test_opencl.dir/build

CMakeFiles/test_opencl.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\test_opencl.dir\cmake_clean.cmake
.PHONY : CMakeFiles/test_opencl.dir/clean

CMakeFiles/test_opencl.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\projects\OpenclProjects\vector_add C:\projects\OpenclProjects\vector_add C:\projects\OpenclProjects\vector_add\cmake-build-debug C:\projects\OpenclProjects\vector_add\cmake-build-debug C:\projects\OpenclProjects\vector_add\cmake-build-debug\CMakeFiles\test_opencl.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_opencl.dir/depend


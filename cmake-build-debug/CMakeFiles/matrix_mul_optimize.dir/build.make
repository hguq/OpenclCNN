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
CMAKE_SOURCE_DIR = C:\projects\OpenclProjects\OPENCL_CNN_INTEGER

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/matrix_mul_optimize.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix_mul_optimize.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix_mul_optimize.dir/flags.make

CMakeFiles/matrix_mul_optimize.dir/main.cpp.obj: CMakeFiles/matrix_mul_optimize.dir/flags.make
CMakeFiles/matrix_mul_optimize.dir/main.cpp.obj: CMakeFiles/matrix_mul_optimize.dir/includes_CXX.rsp
CMakeFiles/matrix_mul_optimize.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrix_mul_optimize.dir/main.cpp.obj"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\matrix_mul_optimize.dir\main.cpp.obj -c C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\main.cpp

CMakeFiles/matrix_mul_optimize.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_mul_optimize.dir/main.cpp.i"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\main.cpp > CMakeFiles\matrix_mul_optimize.dir\main.cpp.i

CMakeFiles/matrix_mul_optimize.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_mul_optimize.dir/main.cpp.s"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\main.cpp -o CMakeFiles\matrix_mul_optimize.dir\main.cpp.s

CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.obj: CMakeFiles/matrix_mul_optimize.dir/flags.make
CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.obj: CMakeFiles/matrix_mul_optimize.dir/includes_CXX.rsp
CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.obj: ../cnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.obj"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\matrix_mul_optimize.dir\cnn.cpp.obj -c C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cnn.cpp

CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.i"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cnn.cpp > CMakeFiles\matrix_mul_optimize.dir\cnn.cpp.i

CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.s"
	C:\programs\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cnn.cpp -o CMakeFiles\matrix_mul_optimize.dir\cnn.cpp.s

# Object files for target matrix_mul_optimize
matrix_mul_optimize_OBJECTS = \
"CMakeFiles/matrix_mul_optimize.dir/main.cpp.obj" \
"CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.obj"

# External object files for target matrix_mul_optimize
matrix_mul_optimize_EXTERNAL_OBJECTS =

matrix_mul_optimize.exe: CMakeFiles/matrix_mul_optimize.dir/main.cpp.obj
matrix_mul_optimize.exe: CMakeFiles/matrix_mul_optimize.dir/cnn.cpp.obj
matrix_mul_optimize.exe: CMakeFiles/matrix_mul_optimize.dir/build.make
matrix_mul_optimize.exe: CMakeFiles/matrix_mul_optimize.dir/linklibs.rsp
matrix_mul_optimize.exe: CMakeFiles/matrix_mul_optimize.dir/objects1.rsp
matrix_mul_optimize.exe: CMakeFiles/matrix_mul_optimize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable matrix_mul_optimize.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\matrix_mul_optimize.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix_mul_optimize.dir/build: matrix_mul_optimize.exe

.PHONY : CMakeFiles/matrix_mul_optimize.dir/build

CMakeFiles/matrix_mul_optimize.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\matrix_mul_optimize.dir\cmake_clean.cmake
.PHONY : CMakeFiles/matrix_mul_optimize.dir/clean

CMakeFiles/matrix_mul_optimize.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\projects\OpenclProjects\OPENCL_CNN_INTEGER C:\projects\OpenclProjects\OPENCL_CNN_INTEGER C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug C:\projects\OpenclProjects\OPENCL_CNN_INTEGER\cmake-build-debug\CMakeFiles\matrix_mul_optimize.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix_mul_optimize.dir/depend


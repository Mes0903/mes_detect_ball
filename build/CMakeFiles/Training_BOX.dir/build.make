# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.24

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\document\GitHub\mes_detect_ball

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\document\GitHub\mes_detect_ball\build

# Include any dependencies generated for this target.
include CMakeFiles/Training_BOX.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Training_BOX.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Training_BOX.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Training_BOX.dir/flags.make

CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj: CMakeFiles/Training_BOX.dir/flags.make
CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj: CMakeFiles/Training_BOX.dir/includes_CXX.rsp
CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj: D:/document/GitHub/mes_detect_ball/src/training_box.cpp
CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj: CMakeFiles/Training_BOX.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\document\GitHub\mes_detect_ball\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj -MF CMakeFiles\Training_BOX.dir\src\training_box.cpp.obj.d -o CMakeFiles\Training_BOX.dir\src\training_box.cpp.obj -c D:\document\GitHub\mes_detect_ball\src\training_box.cpp

CMakeFiles/Training_BOX.dir/src/training_box.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Training_BOX.dir/src/training_box.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\document\GitHub\mes_detect_ball\src\training_box.cpp > CMakeFiles\Training_BOX.dir\src\training_box.cpp.i

CMakeFiles/Training_BOX.dir/src/training_box.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Training_BOX.dir/src/training_box.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\document\GitHub\mes_detect_ball\src\training_box.cpp -o CMakeFiles\Training_BOX.dir\src\training_box.cpp.s

CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj: CMakeFiles/Training_BOX.dir/flags.make
CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj: CMakeFiles/Training_BOX.dir/includes_CXX.rsp
CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj: D:/document/GitHub/mes_detect_ball/lib_def/adaboost_classifier.cpp
CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj: CMakeFiles/Training_BOX.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\document\GitHub\mes_detect_ball\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj -MF CMakeFiles\Training_BOX.dir\lib_def\adaboost_classifier.cpp.obj.d -o CMakeFiles\Training_BOX.dir\lib_def\adaboost_classifier.cpp.obj -c D:\document\GitHub\mes_detect_ball\lib_def\adaboost_classifier.cpp

CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\document\GitHub\mes_detect_ball\lib_def\adaboost_classifier.cpp > CMakeFiles\Training_BOX.dir\lib_def\adaboost_classifier.cpp.i

CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\document\GitHub\mes_detect_ball\lib_def\adaboost_classifier.cpp -o CMakeFiles\Training_BOX.dir\lib_def\adaboost_classifier.cpp.s

CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj: CMakeFiles/Training_BOX.dir/flags.make
CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj: CMakeFiles/Training_BOX.dir/includes_CXX.rsp
CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj: D:/document/GitHub/mes_detect_ball/lib_def/normalize.cpp
CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj: CMakeFiles/Training_BOX.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\document\GitHub\mes_detect_ball\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj -MF CMakeFiles\Training_BOX.dir\lib_def\normalize.cpp.obj.d -o CMakeFiles\Training_BOX.dir\lib_def\normalize.cpp.obj -c D:\document\GitHub\mes_detect_ball\lib_def\normalize.cpp

CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\document\GitHub\mes_detect_ball\lib_def\normalize.cpp > CMakeFiles\Training_BOX.dir\lib_def\normalize.cpp.i

CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\document\GitHub\mes_detect_ball\lib_def\normalize.cpp -o CMakeFiles\Training_BOX.dir\lib_def\normalize.cpp.s

CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj: CMakeFiles/Training_BOX.dir/flags.make
CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj: CMakeFiles/Training_BOX.dir/includes_CXX.rsp
CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj: D:/document/GitHub/mes_detect_ball/lib_def/weak_learner.cpp
CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj: CMakeFiles/Training_BOX.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\document\GitHub\mes_detect_ball\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj -MF CMakeFiles\Training_BOX.dir\lib_def\weak_learner.cpp.obj.d -o CMakeFiles\Training_BOX.dir\lib_def\weak_learner.cpp.obj -c D:\document\GitHub\mes_detect_ball\lib_def\weak_learner.cpp

CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.i"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\document\GitHub\mes_detect_ball\lib_def\weak_learner.cpp > CMakeFiles\Training_BOX.dir\lib_def\weak_learner.cpp.i

CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.s"
	C:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\document\GitHub\mes_detect_ball\lib_def\weak_learner.cpp -o CMakeFiles\Training_BOX.dir\lib_def\weak_learner.cpp.s

# Object files for target Training_BOX
Training_BOX_OBJECTS = \
"CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj" \
"CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj" \
"CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj" \
"CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj"

# External object files for target Training_BOX
Training_BOX_EXTERNAL_OBJECTS =

Training_BOX.exe: CMakeFiles/Training_BOX.dir/src/training_box.cpp.obj
Training_BOX.exe: CMakeFiles/Training_BOX.dir/lib_def/adaboost_classifier.cpp.obj
Training_BOX.exe: CMakeFiles/Training_BOX.dir/lib_def/normalize.cpp.obj
Training_BOX.exe: CMakeFiles/Training_BOX.dir/lib_def/weak_learner.cpp.obj
Training_BOX.exe: CMakeFiles/Training_BOX.dir/build.make
Training_BOX.exe: CMakeFiles/Training_BOX.dir/linklibs.rsp
Training_BOX.exe: CMakeFiles/Training_BOX.dir/objects1.rsp
Training_BOX.exe: CMakeFiles/Training_BOX.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\document\GitHub\mes_detect_ball\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable Training_BOX.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Training_BOX.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Training_BOX.dir/build: Training_BOX.exe
.PHONY : CMakeFiles/Training_BOX.dir/build

CMakeFiles/Training_BOX.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Training_BOX.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Training_BOX.dir/clean

CMakeFiles/Training_BOX.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\document\GitHub\mes_detect_ball D:\document\GitHub\mes_detect_ball D:\document\GitHub\mes_detect_ball\build D:\document\GitHub\mes_detect_ball\build D:\document\GitHub\mes_detect_ball\build\CMakeFiles\Training_BOX.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Training_BOX.dir/depend
